import functools
import gymnasium as gym
import logging
import importlib.util
import os
from typing import (
import ray
from ray.actor import ActorHandle
from ray.exceptions import RayActorError
from ray.rllib.core.learner import LearnerGroup
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.utils.actor_manager import RemoteCallResults
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.offline import get_dataset_and_shards
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.actor_manager import FaultTolerantActorManager
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.policy import validate_policy_id
from ray.rllib.utils.typing import (
@DeveloperAPI
def add_policy(self, policy_id: PolicyID, policy_cls: Optional[Type[Policy]]=None, policy: Optional[Policy]=None, *, observation_space: Optional[gym.spaces.Space]=None, action_space: Optional[gym.spaces.Space]=None, config: Optional[Union['AlgorithmConfig', PartialAlgorithmConfigDict]]=None, policy_state: Optional[PolicyState]=None, policy_mapping_fn: Optional[Callable[[AgentID, EpisodeID], PolicyID]]=None, policies_to_train: Optional[Union[Container[PolicyID], Callable[[PolicyID, Optional[SampleBatchType]], bool]]]=None, module_spec: Optional[SingleAgentRLModuleSpec]=None, workers: Optional[List[Union[EnvRunner, ActorHandle]]]=DEPRECATED_VALUE) -> None:
    """Adds a policy to this WorkerSet's workers or a specific list of workers.

        Args:
            policy_id: ID of the policy to add.
            policy_cls: The Policy class to use for constructing the new Policy.
                Note: Only one of `policy_cls` or `policy` must be provided.
            policy: The Policy instance to add to this WorkerSet. If not None, the
                given Policy object will be directly inserted into the
                local worker and clones of that Policy will be created on all remote
                workers.
                Note: Only one of `policy_cls` or `policy` must be provided.
            observation_space: The observation space of the policy to add.
                If None, try to infer this space from the environment.
            action_space: The action space of the policy to add.
                If None, try to infer this space from the environment.
            config: The config object or overrides for the policy to add.
            policy_state: Optional state dict to apply to the new
                policy instance, right after its construction.
            policy_mapping_fn: An optional (updated) policy mapping function
                to use from here on. Note that already ongoing episodes will
                not change their mapping but will use the old mapping till
                the end of the episode.
            policies_to_train: An optional list of policy IDs to be trained
                or a callable taking PolicyID and SampleBatchType and
                returning a bool (trainable or not?).
                If None, will keep the existing setup in place. Policies,
                whose IDs are not in the list (or for which the callable
                returns False) will not be updated.
            module_spec: In the new RLModule API we need to pass in the module_spec for
                the new module that is supposed to be added. Knowing the policy spec is
                not sufficient.
            workers: A list of EnvRunner/ActorHandles (remote
                EnvRunners) to add this policy to. If defined, will only
                add the given policy to these workers.

        Raises:
            KeyError: If the given `policy_id` already exists in this WorkerSet.
        """
    if self.local_worker() and policy_id in self.local_worker().policy_map:
        raise KeyError(f"Policy ID '{policy_id}' already exists in policy map! Make sure you use a Policy ID that has not been taken yet. Policy IDs that are already in your policy map: {list(self.local_worker().policy_map.keys())}")
    if workers is not DEPRECATED_VALUE:
        deprecation_warning(old='WorkerSet.add_policy(.., workers=..)', help='The `workers` argument to `WorkerSet.add_policy()` is deprecated! Please do not use it anymore.', error=True)
    if (policy_cls is None) == (policy is None):
        raise ValueError('Only one of `policy_cls` or `policy` must be provided to staticmethod: `WorkerSet.add_policy()`!')
    validate_policy_id(policy_id, error=False)
    if policy_cls is not None:
        new_policy_instance_kwargs = dict(policy_id=policy_id, policy_cls=policy_cls, observation_space=observation_space, action_space=action_space, config=config, policy_state=policy_state, policy_mapping_fn=policy_mapping_fn, policies_to_train=list(policies_to_train) if policies_to_train else None, module_spec=module_spec)
    else:
        new_policy_instance_kwargs = dict(policy_id=policy_id, policy_cls=type(policy), observation_space=policy.observation_space, action_space=policy.action_space, config=policy.config, policy_state=policy.get_state(), policy_mapping_fn=policy_mapping_fn, policies_to_train=list(policies_to_train) if policies_to_train else None, module_spec=module_spec)

    def _create_new_policy_fn(worker):
        worker.add_policy(**new_policy_instance_kwargs)
    if self.local_worker() is not None:
        if policy is not None:
            self.local_worker().add_policy(policy_id=policy_id, policy=policy, policy_mapping_fn=policy_mapping_fn, policies_to_train=policies_to_train, module_spec=module_spec)
        else:
            self.local_worker().add_policy(**new_policy_instance_kwargs)
    self.foreach_worker(_create_new_policy_fn, local_worker=False)