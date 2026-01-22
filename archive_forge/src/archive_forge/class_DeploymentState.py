import json
import logging
import math
import os
import random
import time
import traceback
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import ray
from ray import ObjectRef, cloudpickle
from ray.actor import ActorHandle
from ray.exceptions import RayActorError, RayError, RayTaskError, RuntimeEnvSetupError
from ray.serve import metrics
from ray.serve._private import default_impl
from ray.serve._private.autoscaling_metrics import InMemoryMetricsStore
from ray.serve._private.cluster_node_info_cache import ClusterNodeInfoCache
from ray.serve._private.common import (
from ray.serve._private.config import DeploymentConfig
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.deployment_scheduler import (
from ray.serve._private.long_poll import LongPollHost, LongPollNamespace
from ray.serve._private.storage.kv_store import KVStoreBase
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import (
from ray.serve._private.version import DeploymentVersion, VersionedReplica
from ray.serve.generated.serve_pb2 import DeploymentLanguage
from ray.serve.schema import (
from ray.util.placement_group import PlacementGroup
class DeploymentState:
    """Manages the target state and replicas for a single deployment."""
    FORCE_STOP_UNHEALTHY_REPLICAS = RAY_SERVE_FORCE_STOP_UNHEALTHY_REPLICAS

    def __init__(self, id: DeploymentID, controller_name: str, long_poll_host: LongPollHost, deployment_scheduler: DeploymentScheduler, cluster_node_info_cache: ClusterNodeInfoCache, _save_checkpoint_func: Callable):
        self._id = id
        self._controller_name: str = controller_name
        self._long_poll_host: LongPollHost = long_poll_host
        self._deployment_scheduler = deployment_scheduler
        self._cluster_node_info_cache = cluster_node_info_cache
        self._save_checkpoint_func = _save_checkpoint_func
        self._target_state: DeploymentTargetState = DeploymentTargetState.default()
        self._prev_startup_warning: float = time.time()
        self._last_retry: float = 0.0
        self._backoff_time_s: int = 1
        self._replica_constructor_retry_counter: int = 0
        self._replica_constructor_error_msg: Optional[str] = None
        self._replicas: ReplicaStateContainer = ReplicaStateContainer()
        self._curr_status_info: DeploymentStatusInfo = DeploymentStatusInfo(self._id.name, DeploymentStatus.UPDATING, DeploymentStatusTrigger.CONFIG_UPDATE_STARTED)
        self.replica_average_ongoing_requests: Dict[str, float] = dict()
        self.health_check_gauge = metrics.Gauge('serve_deployment_replica_healthy', description='Tracks whether this deployment replica is healthy. 1 means healthy, 0 means unhealthy.', tag_keys=('deployment', 'replica', 'application'))
        self._multiplexed_model_ids_updated = False
        self._last_notified_running_replica_infos: List[RunningReplicaInfo] = []

    def should_autoscale(self) -> bool:
        """
        Check if the deployment is under autoscaling
        """
        return self._target_state.info.autoscaling_policy is not None

    def get_autoscale_metric_lookback_period(self) -> float:
        """
        Return the autoscaling metrics look back period
        """
        return self._target_state.info.autoscaling_policy.config.look_back_period_s

    def get_checkpoint_data(self) -> DeploymentTargetState:
        """
        Return deployment's target state submitted by user's deployment call.
        Should be persisted and outlive current ray cluster.
        """
        return self._target_state

    def recover_target_state_from_checkpoint(self, target_state_checkpoint: DeploymentTargetState):
        logger.info(f'Recovering target state for deployment {self.deployment_name} in application {self.app_name} from checkpoint.')
        self._target_state = target_state_checkpoint

    def recover_current_state_from_replica_actor_names(self, replica_actor_names: List[str]):
        """Recover deployment state from live replica actors found in the cluster."""
        assert self._target_state is not None, 'Target state should be recovered successfully first before recovering current state from replica actor names.'
        logger.info(f"Recovering current state for deployment '{self.deployment_name}' in application '{self.app_name}' from {len(replica_actor_names)} total actors.")
        for replica_actor_name in replica_actor_names:
            replica_name: ReplicaName = ReplicaName.from_str(replica_actor_name)
            new_deployment_replica = DeploymentReplica(self._controller_name, replica_name.replica_tag, replica_name.deployment_id, self._target_state.version)
            new_deployment_replica.recover()
            self._replicas.add(ReplicaState.RECOVERING, new_deployment_replica)
            self._deployment_scheduler.on_replica_recovering(replica_name.deployment_id, replica_name.replica_tag)
            logger.debug(f'RECOVERING replica: {new_deployment_replica.replica_tag}, deployment: {self.deployment_name}, application: {self.app_name}.')

    @property
    def target_info(self) -> DeploymentInfo:
        return self._target_state.info

    @property
    def curr_status_info(self) -> DeploymentStatusInfo:
        return self._curr_status_info

    @property
    def deployment_name(self) -> str:
        return self._id.name

    @property
    def app_name(self) -> str:
        return self._id.app

    def get_running_replica_infos(self) -> List[RunningReplicaInfo]:
        return [replica.get_running_replica_info(self._cluster_node_info_cache) for replica in self._replicas.get([ReplicaState.RUNNING])]

    def get_active_node_ids(self) -> Set[str]:
        """Get the node ids of all running replicas in this deployment.

        This is used to determine which node has replicas. Only nodes with replicas and
        head node should have active proxies.
        """
        active_states = [ReplicaState.STARTING, ReplicaState.UPDATING, ReplicaState.RECOVERING, ReplicaState.RUNNING]
        return {replica.actor_node_id for replica in self._replicas.get(active_states) if replica.actor_node_id is not None}

    def list_replica_details(self) -> List[ReplicaDetails]:
        return [replica.actor_details for replica in self._replicas.get()]

    def notify_running_replicas_changed(self) -> None:
        running_replica_infos = self.get_running_replica_infos()
        if set(self._last_notified_running_replica_infos) == set(running_replica_infos) and (not self._multiplexed_model_ids_updated):
            return
        self._long_poll_host.notify_changed((LongPollNamespace.RUNNING_REPLICAS, self._id), running_replica_infos)
        self._long_poll_host.notify_changed((LongPollNamespace.RUNNING_REPLICAS, self._id.name), running_replica_infos)
        self._last_notified_running_replica_infos = running_replica_infos
        self._multiplexed_model_ids_updated = False

    def _set_target_state_deleting(self) -> None:
        """Set the target state for the deployment to be deleted."""
        target_state = DeploymentTargetState.create(info=self._target_state.info, target_num_replicas=0, deleting=True)
        self._save_checkpoint_func(writeahead_checkpoints={self._id: target_state})
        self._target_state = target_state
        self._curr_status_info = DeploymentStatusInfo(self.deployment_name, DeploymentStatus.UPDATING, status_trigger=DeploymentStatusTrigger.DELETING)
        app_msg = f" in application '{self.app_name}'" if self.app_name else ''
        logger.info(f'Deleting deployment {self.deployment_name}{app_msg}', extra={'log_to_stderr': False})

    def _set_target_state(self, target_info: DeploymentInfo, target_num_replicas: int, status_trigger: DeploymentStatusTrigger, allow_scaling_statuses: bool) -> None:
        """Set the target state for the deployment to the provided info.

        Args:
            target_info: The info with which to set the target state.
            target_num_replicas: The number of replicas that this deployment
                should attempt to run.
            status_trigger: The driver that triggered this change of state.
            allow_scaling_statuses: Whether to allow this method
                to set the status to UPSCALING/DOWNSCALING or not.
        """
        new_target_state = DeploymentTargetState.create(target_info, target_num_replicas, deleting=False)
        self._save_checkpoint_func(writeahead_checkpoints={self._id: new_target_state})
        if self._target_state.version == new_target_state.version:
            if self._target_state.version.deployment_config.autoscaling_config != new_target_state.version.deployment_config.autoscaling_config:
                ServeUsageTag.AUTOSCALING_CONFIG_LIGHTWEIGHT_UPDATED.record('True')
            elif self._target_state.version.deployment_config.num_replicas != new_target_state.version.deployment_config.num_replicas:
                ServeUsageTag.NUM_REPLICAS_LIGHTWEIGHT_UPDATED.record('True')
        if new_target_state.is_scaled_copy_of(self._target_state):
            curr_num_replicas = self._target_state.target_num_replicas
            new_num_replicas = new_target_state.target_num_replicas
            num_replicas_changed = curr_num_replicas != new_num_replicas
            if allow_scaling_statuses and num_replicas_changed:
                scaling_direction = DeploymentStatus.UPSCALING if new_num_replicas > curr_num_replicas else DeploymentStatus.DOWNSCALING
                self._curr_status_info = self._curr_status_info.update(status=scaling_direction, status_trigger=status_trigger, message=f'{scaling_direction.capitalize()} from {curr_num_replicas} to {new_num_replicas} replicas.')
        else:
            self._curr_status_info = self._curr_status_info.update(status=DeploymentStatus.UPDATING, status_trigger=status_trigger)
        self._target_state = new_target_state

    def deploy(self, deployment_info: DeploymentInfo) -> bool:
        """Deploy the deployment.

        If the deployment already exists with the same version, config,
        target_capacity, and target_capacity_direction,
        this method returns False.

        Returns:
            bool: Whether or not the deployment is being updated.
        """
        curr_deployment_info = self._target_state.info
        if curr_deployment_info is not None:
            if not self._target_state.deleting:
                deployment_info.start_time_ms = curr_deployment_info.start_time_ms
            deployment_settings_changed = self._target_state.deleting or curr_deployment_info.deployment_config != deployment_info.deployment_config or curr_deployment_info.replica_config.ray_actor_options != deployment_info.replica_config.ray_actor_options or (deployment_info.version is None) or (curr_deployment_info.version != deployment_info.version)
            target_capacity_changed = curr_deployment_info.target_capacity != deployment_info.target_capacity or curr_deployment_info.target_capacity_direction != deployment_info.target_capacity_direction
        else:
            deployment_settings_changed = True
            target_capacity_changed = True
        if not deployment_settings_changed and (not target_capacity_changed):
            return False
        autoscaling_policy = deployment_info.autoscaling_policy
        if autoscaling_policy is not None:
            if deployment_settings_changed and autoscaling_policy.config.initial_replicas is not None:
                target_num_replicas = get_capacity_adjusted_num_replicas(autoscaling_policy.config.initial_replicas, deployment_info.target_capacity)
            else:
                target_num_replicas = autoscaling_policy.apply_bounds(self._target_state.target_num_replicas, deployment_info.target_capacity, deployment_info.target_capacity_direction)
        else:
            target_num_replicas = get_capacity_adjusted_num_replicas(deployment_info.deployment_config.num_replicas, deployment_info.target_capacity)
        allow_scaling_statuses = self.curr_status_info.status is not DeploymentStatus.UPDATING
        self._set_target_state(deployment_info, target_num_replicas=target_num_replicas, status_trigger=DeploymentStatusTrigger.CONFIG_UPDATE_STARTED, allow_scaling_statuses=allow_scaling_statuses)
        logger.info(f"Deploying new version of deployment {self.deployment_name} in application '{self.app_name}'. Setting initial target number of replicas to {target_num_replicas}.")
        self._replica_constructor_retry_counter = 0
        self._backoff_time_s = 1
        return True

    def get_replica_current_ongoing_requests(self) -> List[float]:
        """Return list of replica average ongoing requests.

        The length of list indicate the number of replicas.
        """
        running_replicas = self._replicas.get([ReplicaState.RUNNING])
        current_num_ongoing_requests = []
        for replica in running_replicas:
            replica_tag = replica.replica_tag
            if replica_tag in self.replica_average_ongoing_requests:
                current_num_ongoing_requests.append(self.replica_average_ongoing_requests[replica_tag])
        return current_num_ongoing_requests

    def autoscale(self, current_handle_queued_queries: int) -> int:
        """Autoscale the deployment based on metrics.

        Args:
            current_handle_queued_queries: The number of handle queued queries,
                if there are multiple handles, the max number of queries at
                a single handle should be passed in
        """
        if self._target_state.deleting:
            return
        current_num_ongoing_requests = self.get_replica_current_ongoing_requests()
        autoscaling_policy = self._target_state.info.autoscaling_policy
        decision_num_replicas = autoscaling_policy.get_decision_num_replicas(curr_target_num_replicas=self._target_state.target_num_replicas, current_num_ongoing_requests=current_num_ongoing_requests, current_handle_queued_queries=current_handle_queued_queries, target_capacity=self._target_state.info.target_capacity, target_capacity_direction=self._target_state.info.target_capacity_direction)
        if decision_num_replicas == self._target_state.target_num_replicas:
            return
        logger.info(f'Autoscaling replicas for deployment {self.deployment_name} in application {self.app_name} to {decision_num_replicas}. current_num_ongoing_requests: {current_num_ongoing_requests}, current handle queued queries: {current_handle_queued_queries}.')
        new_info = copy(self._target_state.info)
        new_info.version = self._target_state.version.code_version
        allow_scaling_statuses = self._is_within_autoscaling_bounds() or self.curr_status_info.status_trigger != DeploymentStatusTrigger.CONFIG_UPDATE_STARTED
        self._set_target_state(new_info, decision_num_replicas, status_trigger=DeploymentStatusTrigger.AUTOSCALING, allow_scaling_statuses=allow_scaling_statuses)

    def _is_within_autoscaling_bounds(self) -> bool:
        """Whether or not this deployment is within the autoscaling bounds.

        This method should only be used for autoscaling deployments. It raises
        an assertion error otherwise.

        Returns: True if the number of running replicas for the current
            deployment version is within the autoscaling bounds. False
            otherwise.
        """
        target_version = self._target_state.version
        num_replicas_running_at_target_version = self._replicas.count(states=[ReplicaState.RUNNING], version=target_version)
        autoscaling_policy = self._target_state.info.autoscaling_policy
        assert autoscaling_policy is not None
        lower_bound = autoscaling_policy.get_current_lower_bound(self._target_state.info.target_capacity, self._target_state.info.target_capacity_direction)
        upper_bound = get_capacity_adjusted_num_replicas(autoscaling_policy.config.max_replicas, self._target_state.info.target_capacity)
        return lower_bound <= num_replicas_running_at_target_version <= upper_bound

    def delete(self) -> None:
        if not self._target_state.deleting:
            self._set_target_state_deleting()

    def _stop_or_update_outdated_version_replicas(self, max_to_stop=math.inf) -> bool:
        """Stop or update replicas with outdated versions.

        Stop replicas with versions that require the actor to be restarted, and
        reconfigure replicas that require refreshing deployment config values.

        Args:
            max_to_stop: max number of replicas to stop, by default,
            it will stop all replicas with outdated version.
        """
        replicas_to_update = self._replicas.pop(exclude_version=self._target_state.version, states=[ReplicaState.STARTING, ReplicaState.RUNNING])
        replicas_changed = False
        code_version_changes = 0
        reconfigure_changes = 0
        for replica in replicas_to_update:
            if code_version_changes + reconfigure_changes >= max_to_stop:
                self._replicas.add(replica.actor_details.state, replica)
            elif replica.version.requires_actor_restart(self._target_state.version):
                code_version_changes += 1
                graceful_stop = replica.actor_details.state == ReplicaState.RUNNING
                self._stop_replica(replica, graceful_stop=graceful_stop)
                replicas_changed = True
            elif replica.actor_details.state == ReplicaState.RUNNING:
                reconfigure_changes += 1
                if replica.version.requires_long_poll_broadcast(self._target_state.version):
                    replicas_changed = True
                actor_updating = replica.reconfigure(self._target_state.version)
                if actor_updating:
                    self._replicas.add(ReplicaState.UPDATING, replica)
                else:
                    self._replicas.add(ReplicaState.RUNNING, replica)
                logger.debug(f'Adding UPDATING to replica_tag: {replica.replica_tag}, deployment_name: {self.deployment_name}, app_name: {self.app_name}')
            else:
                self._replicas.add(replica.actor_details.state, replica)
        if code_version_changes > 0:
            logger.info(f"Stopping {code_version_changes} replicas of deployment '{self.deployment_name}' in application '{self.app_name}' with outdated versions.")
        if reconfigure_changes > 0:
            logger.info(f"Updating {reconfigure_changes} replicas of deployment '{self.deployment_name}' in application '{self.app_name}' with outdated deployment configs.")
            ServeUsageTag.USER_CONFIG_LIGHTWEIGHT_UPDATED.record('True')
        return replicas_changed

    def _check_and_stop_outdated_version_replicas(self) -> bool:
        """Stops replicas with outdated versions to implement rolling updates.

        This includes both explicit code version updates and changes to the
        user_config.

        Returns whether any replicas were stopped.
        """
        if self._target_state.target_num_replicas == 0:
            return False
        old_running_replicas = self._replicas.count(exclude_version=self._target_state.version, states=[ReplicaState.STARTING, ReplicaState.UPDATING, ReplicaState.RUNNING])
        old_stopping_replicas = self._replicas.count(exclude_version=self._target_state.version, states=[ReplicaState.STOPPING])
        new_running_replicas = self._replicas.count(version=self._target_state.version, states=[ReplicaState.RUNNING])
        if self._target_state.target_num_replicas < old_running_replicas + old_stopping_replicas:
            return False
        pending_replicas = self._target_state.target_num_replicas - new_running_replicas - old_running_replicas
        rollout_size = max(int(0.2 * self._target_state.target_num_replicas), 1)
        max_to_stop = max(rollout_size - pending_replicas, 0)
        return self._stop_or_update_outdated_version_replicas(max_to_stop)

    def _scale_deployment_replicas(self) -> Tuple[List[ReplicaSchedulingRequest], DeploymentDownscaleRequest]:
        """Scale the given deployment to the number of replicas."""
        assert self._target_state.target_num_replicas >= 0, 'Target number of replicas must be greater than or equal to 0.'
        upscale = []
        downscale = None
        self._check_and_stop_outdated_version_replicas()
        current_replicas = self._replicas.count(states=[ReplicaState.STARTING, ReplicaState.UPDATING, ReplicaState.RUNNING])
        recovering_replicas = self._replicas.count(states=[ReplicaState.RECOVERING])
        delta_replicas = self._target_state.target_num_replicas - current_replicas - recovering_replicas
        if delta_replicas == 0:
            return (upscale, downscale)
        elif delta_replicas > 0:
            stopping_replicas = self._replicas.count(states=[ReplicaState.STOPPING])
            to_add = max(delta_replicas - stopping_replicas, 0)
            if to_add > 0:
                failed_to_start_threshold = min(MAX_DEPLOYMENT_CONSTRUCTOR_RETRY_COUNT, self._target_state.target_num_replicas * 3)
                if self._replica_constructor_retry_counter >= failed_to_start_threshold:
                    if time.time() - self._last_retry < self._backoff_time_s + random.uniform(0, 3):
                        return (upscale, downscale)
                self._last_retry = time.time()
                logger.info(f"Adding {to_add} replica{('s' if to_add > 1 else '')} to deployment {self.deployment_name} in application '{self.app_name}'.")
                for _ in range(to_add):
                    replica_name = ReplicaName(self.app_name, self.deployment_name, get_random_letters())
                    new_deployment_replica = DeploymentReplica(self._controller_name, replica_name.replica_tag, self._id, self._target_state.version)
                    upscale.append(new_deployment_replica.start(self._target_state.info))
                    self._replicas.add(ReplicaState.STARTING, new_deployment_replica)
                    logger.debug(f"Adding STARTING to replica_tag: {replica_name}, deployment: '{self.deployment_name}', application: '{self.app_name}'")
        elif delta_replicas < 0:
            to_remove = -delta_replicas
            logger.info(f"Removing {to_remove} replica{('s' if to_remove > 1 else '')} from deployment '{self.deployment_name}' in application '{self.app_name}'.")
            downscale = DeploymentDownscaleRequest(deployment_id=self._id, num_to_stop=to_remove)
        return (upscale, downscale)

    def _check_curr_status(self) -> Tuple[bool, bool]:
        """Check the current deployment status.

        Checks the difference between the target vs. running replica count for
        the target version.

        This will update the current deployment status depending on the state
        of the replicas.

        Returns (deleted, any_replicas_recovering).
        """
        target_version = self._target_state.version
        any_replicas_recovering = self._replicas.count(states=[ReplicaState.RECOVERING]) > 0
        all_running_replica_cnt = self._replicas.count(states=[ReplicaState.RUNNING])
        running_at_target_version_replica_cnt = self._replicas.count(states=[ReplicaState.RUNNING], version=target_version)
        failed_to_start_count = self._replica_constructor_retry_counter
        failed_to_start_threshold = min(MAX_DEPLOYMENT_CONSTRUCTOR_RETRY_COUNT, self._target_state.target_num_replicas * 3)
        if failed_to_start_count >= failed_to_start_threshold and failed_to_start_threshold != 0:
            if running_at_target_version_replica_cnt > 0:
                self._replica_constructor_retry_counter = -1
            else:
                self._curr_status_info = self._curr_status_info.update(status=DeploymentStatus.UNHEALTHY, status_trigger=DeploymentStatusTrigger.REPLICA_STARTUP_FAILED, message=f'The deployment failed to start {failed_to_start_count} times in a row. This may be due to a problem with its constructor or initial health check failing. See controller logs for details. Retrying after {self._backoff_time_s} seconds. Error:\n{self._replica_constructor_error_msg}')
                return (False, any_replicas_recovering)
        if self._replicas.count(states=[ReplicaState.STARTING, ReplicaState.UPDATING, ReplicaState.RECOVERING, ReplicaState.STOPPING]) == 0:
            if self._target_state.deleting and all_running_replica_cnt == 0:
                return (True, any_replicas_recovering)
            if self._target_state.target_num_replicas == running_at_target_version_replica_cnt and running_at_target_version_replica_cnt == all_running_replica_cnt:
                if self._curr_status_info.status == DeploymentStatus.UPSCALING:
                    status_trigger = DeploymentStatusTrigger.UPSCALE_COMPLETED
                elif self._curr_status_info.status == DeploymentStatus.DOWNSCALING:
                    status_trigger = DeploymentStatusTrigger.DOWNSCALE_COMPLETED
                elif self._curr_status_info.status == DeploymentStatus.UPDATING and self._curr_status_info.status_trigger == DeploymentStatusTrigger.CONFIG_UPDATE_STARTED:
                    status_trigger = DeploymentStatusTrigger.CONFIG_UPDATE_COMPLETED
                elif self._curr_status_info.status == DeploymentStatus.UNHEALTHY:
                    status_trigger = DeploymentStatusTrigger.UNSPECIFIED
                else:
                    status_trigger = self._curr_status_info.status_trigger
                self._curr_status_info = self._curr_status_info.update(status=DeploymentStatus.HEALTHY, status_trigger=status_trigger)
                return (False, any_replicas_recovering)
        return (False, any_replicas_recovering)

    def _check_startup_replicas(self, original_state: ReplicaState, stop_on_slow=False) -> List[Tuple[DeploymentReplica, ReplicaStartupStatus]]:
        """
        Common helper function for startup actions tracking and status
        transition: STARTING, UPDATING and RECOVERING.

        Args:
            stop_on_slow: If we consider a replica failed upon observing it's
                slow to reach running state.
        """
        slow_replicas = []
        replicas_failed = False
        for replica in self._replicas.pop(states=[original_state]):
            start_status, error_msg = replica.check_started()
            if start_status == ReplicaStartupStatus.SUCCEEDED:
                self._replicas.add(ReplicaState.RUNNING, replica)
                self._deployment_scheduler.on_replica_running(self._id, replica.replica_tag, replica.actor_node_id)
                logger.info(f'Replica {replica.replica_tag} started successfully on node {replica.actor_node_id}.', extra={'log_to_stderr': False})
            elif start_status == ReplicaStartupStatus.FAILED:
                if self._replica_constructor_retry_counter >= 0:
                    self._replica_constructor_retry_counter += 1
                    self._replica_constructor_error_msg = error_msg
                replicas_failed = True
                self._stop_replica(replica)
            elif start_status in [ReplicaStartupStatus.PENDING_ALLOCATION, ReplicaStartupStatus.PENDING_INITIALIZATION]:
                is_slow = time.time() - replica._start_time > SLOW_STARTUP_WARNING_S
                if is_slow:
                    slow_replicas.append((replica, start_status))
                if is_slow and stop_on_slow:
                    self._stop_replica(replica, graceful_stop=False)
                else:
                    self._replicas.add(original_state, replica)
        failed_to_start_threshold = min(MAX_DEPLOYMENT_CONSTRUCTOR_RETRY_COUNT, self._target_state.target_num_replicas * 3)
        if replicas_failed and self._replica_constructor_retry_counter > failed_to_start_threshold:
            self._backoff_time_s = min(EXPONENTIAL_BACKOFF_FACTOR * self._backoff_time_s, MAX_BACKOFF_TIME_S)
        return slow_replicas

    def stop_replicas(self, replicas_to_stop) -> None:
        for replica in self._replicas.pop():
            if replica.replica_tag in replicas_to_stop:
                self._stop_replica(replica)
            else:
                self._replicas.add(replica.actor_details.state, replica)

    def _stop_replica(self, replica: VersionedReplica, graceful_stop=True):
        """Stop replica
        1. Stop the replica.
        2. Change the replica into stopping state.
        3. Set the health replica stats to 0.
        """
        logger.debug(f'Adding STOPPING to replica_tag: {replica}, deployment_name: {self.deployment_name}, app_name: {self.app_name}')
        replica.stop(graceful=graceful_stop)
        self._replicas.add(ReplicaState.STOPPING, replica)
        self._deployment_scheduler.on_replica_stopping(self._id, replica.replica_tag)
        self.health_check_gauge.set(0, tags={'deployment': self.deployment_name, 'replica': replica.replica_tag, 'application': self.app_name})

    def _check_and_update_replicas(self):
        """
        Check current state of all DeploymentReplica being tracked, and compare
        with state container from previous update() cycle to see if any state
        transition happened.
        """
        for replica in self._replicas.pop(states=[ReplicaState.RUNNING]):
            if replica.check_health():
                self._replicas.add(ReplicaState.RUNNING, replica)
                self.health_check_gauge.set(1, tags={'deployment': self.deployment_name, 'replica': replica.replica_tag, 'application': self.app_name})
            else:
                logger.warning(f"Replica {replica.replica_tag} of deployment {self.deployment_name} in application '{self.app_name}' failed health check, stopping it.")
                self.health_check_gauge.set(0, tags={'deployment': self.deployment_name, 'replica': replica.replica_tag, 'application': self.app_name})
                self._stop_replica(replica, graceful_stop=not self.FORCE_STOP_UNHEALTHY_REPLICAS)
                if replica.version == self._target_state.version:
                    self._curr_status_info = self._curr_status_info.update(status=DeploymentStatus.UNHEALTHY, status_trigger=DeploymentStatusTrigger.HEALTH_CHECK_FAILED, message="A replica's health check failed. This deployment will be UNHEALTHY until the replica recovers or a new deploy happens.")
        slow_start_replicas = []
        slow_start = self._check_startup_replicas(ReplicaState.STARTING)
        slow_update = self._check_startup_replicas(ReplicaState.UPDATING)
        slow_recover = self._check_startup_replicas(ReplicaState.RECOVERING, stop_on_slow=True)
        slow_start_replicas = slow_start + slow_update + slow_recover
        if len(slow_start_replicas) and time.time() - self._prev_startup_warning > SLOW_STARTUP_WARNING_PERIOD_S:
            pending_allocation = []
            pending_initialization = []
            for replica, startup_status in slow_start_replicas:
                if startup_status == ReplicaStartupStatus.PENDING_ALLOCATION:
                    pending_allocation.append(replica)
                if startup_status == ReplicaStartupStatus.PENDING_INITIALIZATION:
                    pending_initialization.append(replica)
            if len(pending_allocation) > 0:
                required, available = pending_allocation[0].resource_requirements()
                message = f"Deployment '{self.deployment_name}' in application '{self.app_name}' {len(pending_allocation)} replicas that have taken more than {SLOW_STARTUP_WARNING_S}s to be scheduled. This may be due to waiting for the cluster to auto-scale or for a runtime environment to be installed. Resources required for each replica: {required}, total resources available: {available}. Use `ray status` for more details."
                logger.warning(message)
                if _SCALING_LOG_ENABLED:
                    print_verbose_scaling_log()
                if self._curr_status_info.status != DeploymentStatus.UNHEALTHY:
                    self._curr_status_info = self._curr_status_info.update(message=message)
            if len(pending_initialization) > 0:
                message = f"Deployment '{self.deployment_name}' in application '{self.app_name}' has {len(pending_initialization)} replicas that have taken more than {SLOW_STARTUP_WARNING_S}s to initialize. This may be caused by a slow __init__ or reconfigure method."
                logger.warning(message)
                if self._curr_status_info.status != DeploymentStatus.UNHEALTHY:
                    self._curr_status_info = self._curr_status_info.update(message=message)
            self._prev_startup_warning = time.time()
        for replica in self._replicas.pop(states=[ReplicaState.STOPPING]):
            stopped = replica.check_stopped()
            if not stopped:
                self._replicas.add(ReplicaState.STOPPING, replica)
            else:
                logger.info(f'Replica {replica.replica_tag} is stopped.')
                if replica.replica_tag in self.replica_average_ongoing_requests:
                    del self.replica_average_ongoing_requests[replica.replica_tag]

    def _stop_replicas_on_draining_nodes(self):
        draining_nodes = self._cluster_node_info_cache.get_draining_node_ids()
        for replica in self._replicas.pop(states=[ReplicaState.UPDATING, ReplicaState.RUNNING]):
            if replica.actor_node_id in draining_nodes:
                state = replica._actor_details.state
                logger.info(f"Stopping replica {replica.replica_tag} (currently {state}) of deployment '{self.deployment_name}' in application '{self.app_name}' on draining node {replica.actor_node_id}.")
                self._stop_replica(replica, graceful_stop=True)
            else:
                self._replicas.add(replica.actor_details.state, replica)

    def update(self) -> DeploymentStateUpdateResult:
        """Attempts to reconcile this deployment to match its goal state.

        This is an asynchronous call; it's expected to be called repeatedly.

        Also updates the internal DeploymentStatusInfo based on the current
        state of the system.
        """
        deleted, any_replicas_recovering = (False, False)
        upscale = []
        downscale = None
        try:
            self._check_and_update_replicas()
            self._stop_replicas_on_draining_nodes()
            upscale, downscale = self._scale_deployment_replicas()
            deleted, any_replicas_recovering = self._check_curr_status()
        except Exception:
            logger.exception('Exception occurred trying to update deployment state:\n' + traceback.format_exc())
            self._curr_status_info = self._curr_status_info.update(status=DeploymentStatus.UNHEALTHY, status_trigger=DeploymentStatusTrigger.INTERNAL_ERROR, message=f'Failed to update deployment:\n{traceback.format_exc()}')
        return DeploymentStateUpdateResult(deleted=deleted, any_replicas_recovering=any_replicas_recovering, upscale=upscale, downscale=downscale)

    def record_autoscaling_metrics(self, replica_tag: str, window_avg: float) -> None:
        """Records average ongoing requests at replicas."""
        self.replica_average_ongoing_requests[replica_tag] = window_avg

    def record_multiplexed_model_ids(self, replica_name: str, multiplexed_model_ids: List[str]) -> None:
        """Records the multiplexed model IDs of a replica.

        Args:
            replica_name: Name of the replica.
            multiplexed_model_ids: List of model IDs that replica is serving.
        """
        for replica in self._replicas.get():
            if replica.replica_tag == replica_name:
                replica.record_multiplexed_model_ids(multiplexed_model_ids)
                self._multiplexed_model_ids_updated = True
                return
        logger.warn(f'Replia {replica_name} not found in deployment {self.deployment_name} in application {self.app_name}')

    def _stop_one_running_replica_for_testing(self):
        running_replicas = self._replicas.pop(states=[ReplicaState.RUNNING])
        replica_to_stop = running_replicas.pop()
        replica_to_stop.stop(graceful=False)
        self._replicas.add(ReplicaState.STOPPING, replica_to_stop)
        for replica in running_replicas:
            self._replicas.add(ReplicaState.RUNNING, replica)