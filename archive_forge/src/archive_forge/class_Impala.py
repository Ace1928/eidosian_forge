import copy
import dataclasses
from functools import partial
import logging
import platform
import queue
import random
from typing import Callable, List, Optional, Set, Tuple, Type, Union
import numpy as np
import tree  # pip install dm_tree
import ray
from ray import ObjectRef
from ray.rllib import SampleBatch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.impala.impala_learner import (
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.evaluation.worker_set import handle_remote_call_result_errors
from ray.rllib.execution.buffers.mixin_replay_buffer import MixInMultiAgentReplayBuffer
from ray.rllib.execution.learner_thread import LearnerThread
from ray.rllib.execution.multi_gpu_learner_thread import MultiGPULearnerThread
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import concat_samples
from ray.rllib.utils.actor_manager import (
from ray.rllib.utils.actors import create_colocated_actors
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import ALL_MODULES
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import ReplayMode
from ray.rllib.utils.replay_buffers.replay_buffer import _ALL_POLICIES
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.typing import (
from ray.tune.execution.placement_groups import PlacementGroupFactory
class Impala(Algorithm):
    """Importance weighted actor/learner architecture (IMPALA) Algorithm

    == Overview of data flow in IMPALA ==
    1. Policy evaluation in parallel across `num_workers` actors produces
       batches of size `rollout_fragment_length * num_envs_per_worker`.
    2. If enabled, the replay buffer stores and produces batches of size
       `rollout_fragment_length * num_envs_per_worker`.
    3. If enabled, the minibatch ring buffer stores and replays batches of
       size `train_batch_size` up to `num_sgd_iter` times per batch.
    4. The learner thread executes data parallel SGD across `num_gpus` GPUs
       on batches of size `train_batch_size`.
    """

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return ImpalaConfig()

    @classmethod
    @override(Algorithm)
    def get_default_policy_class(cls, config: AlgorithmConfig) -> Optional[Type[Policy]]:
        if not config['vtrace']:
            raise ValueError('IMPALA with the learner API does not support non-VTrace ')
        if config['framework'] == 'torch':
            if config['vtrace']:
                from ray.rllib.algorithms.impala.impala_torch_policy import ImpalaTorchPolicy
                return ImpalaTorchPolicy
            else:
                from ray.rllib.algorithms.a3c.a3c_torch_policy import A3CTorchPolicy
                return A3CTorchPolicy
        elif config['framework'] == 'tf':
            if config['vtrace']:
                from ray.rllib.algorithms.impala.impala_tf_policy import ImpalaTF1Policy
                return ImpalaTF1Policy
            else:
                from ray.rllib.algorithms.a3c.a3c_tf_policy import A3CTFPolicy
                return A3CTFPolicy
        elif config['vtrace']:
            from ray.rllib.algorithms.impala.impala_tf_policy import ImpalaTF2Policy
            return ImpalaTF2Policy
        else:
            from ray.rllib.algorithms.a3c.a3c_tf_policy import A3CTFPolicy
            return A3CTFPolicy

    @override(Algorithm)
    def setup(self, config: AlgorithmConfig):
        super().setup(config)
        self.batches_to_place_on_learner = []
        self.batch_being_built = []
        if self.config.num_aggregation_workers > 0:
            localhost = platform.node()
            assert localhost != '', 'ERROR: Cannot determine local node name! `platform.node()` returned empty string.'
            all_co_located = create_colocated_actors(actor_specs=[(AggregatorWorker, [self.config], {}, self.config.num_aggregation_workers)], node=localhost)
            aggregator_workers = [actor for actor_groups in all_co_located for actor in actor_groups]
            self._aggregator_actor_manager = FaultTolerantActorManager(aggregator_workers, max_remote_requests_in_flight_per_actor=self.config.max_requests_in_flight_per_aggregator_worker)
            self._timeout_s_aggregator_manager = self.config.timeout_s_aggregator_manager
        else:
            self.local_mixin_buffer = MixInMultiAgentReplayBuffer(capacity=self.config.replay_buffer_num_slots if self.config.replay_buffer_num_slots > 0 else 1, replay_ratio=self.config.get_replay_ratio(), replay_mode=ReplayMode.LOCKSTEP)
            self._aggregator_actor_manager = None
        self._results = {}
        self._timeout_s_sampler_manager = self.config.timeout_s_sampler_manager
        if not self.config._enable_new_api_stack:
            self._learner_thread = make_learner_thread(self.workers.local_worker(), self.config)
            self._learner_thread.start()

    @override(Algorithm)
    def training_step(self) -> ResultDict:
        if not self.config._enable_new_api_stack and (not self._learner_thread.is_alive()):
            raise RuntimeError('The learner thread died while training!')
        use_tree_aggregation = self._aggregator_actor_manager and self._aggregator_actor_manager.num_healthy_actors() > 0
        unprocessed_sample_batches = self.get_samples_from_workers(return_object_refs=use_tree_aggregation)
        workers_that_need_updates = {worker_id for worker_id, _ in unprocessed_sample_batches}
        if use_tree_aggregation:
            batches = self.process_experiences_tree_aggregation(unprocessed_sample_batches)
        else:
            batches = self.process_experiences_directly(unprocessed_sample_batches)
        for batch in batches:
            self._counters[NUM_ENV_STEPS_SAMPLED] += batch.count
            self._counters[NUM_AGENT_STEPS_SAMPLED] += batch.agent_steps()
        self.concatenate_batches_and_pre_queue(batches)
        if self.config._enable_new_api_stack:
            train_results = self.learn_on_processed_samples()
            additional_results = self.learner_group.additional_update(module_ids_to_update=set(train_results.keys()) - {ALL_MODULES}, timestep=self._counters[NUM_ENV_STEPS_TRAINED if self.config.count_steps_by == 'env_steps' else NUM_AGENT_STEPS_TRAINED], **self._get_additional_update_kwargs(train_results))
            for key, res in additional_results.items():
                if key in train_results:
                    train_results[key].update(res)
        else:
            self.place_processed_samples_on_learner_thread_queue()
            train_results = self.process_trained_results()
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            if self.config._enable_new_api_stack:
                if train_results:
                    pids = list(set(train_results.keys()) - {ALL_MODULES})
                    self.update_workers_from_learner_group(workers_that_need_updates=workers_that_need_updates, policy_ids=pids)
            else:
                pids = list(train_results.keys())
                self.update_workers_if_necessary(workers_that_need_updates=workers_that_need_updates, policy_ids=pids)
        if self._aggregator_actor_manager:
            self._aggregator_actor_manager.probe_unhealthy_actors(timeout_seconds=self.config.worker_health_probe_timeout_s, mark_healthy=True)
        if self.config._enable_new_api_stack:
            if train_results:
                self._results = train_results
            return self._results
        else:
            return train_results

    @classmethod
    @override(Algorithm)
    def default_resource_request(cls, config: Union[AlgorithmConfig, PartialAlgorithmConfigDict]):
        if isinstance(config, AlgorithmConfig):
            cf: ImpalaConfig = config
        else:
            cf: ImpalaConfig = cls.get_default_config().update_from_dict(config)
        eval_config = cf.get_evaluation_config_object()
        bundles = [{'CPU': cf.num_cpus_for_local_worker + cf.num_aggregation_workers, 'GPU': 0 if cf._fake_gpus else cf.num_gpus}] + [{'CPU': cf.num_cpus_per_worker, 'GPU': cf.num_gpus_per_worker, **cf.custom_resources_per_worker} for _ in range(cf.num_rollout_workers)] + ([{'CPU': eval_config.num_cpus_per_worker, 'GPU': eval_config.num_gpus_per_worker, **eval_config.custom_resources_per_worker} for _ in range(cf.evaluation_num_workers)] if cf.evaluation_interval else [])
        if cf._enable_new_api_stack:
            learner_bundles = cls._get_learner_bundles(cf)
            bundles += learner_bundles
        return PlacementGroupFactory(bundles=bundles, strategy=cf.placement_strategy)

    def concatenate_batches_and_pre_queue(self, batches: List[SampleBatch]):
        """Concatenate batches that are being returned from rollout workers

        Args:
            batches: batches of experiences from rollout workers

        """

        def aggregate_into_larger_batch():
            if sum((b.count for b in self.batch_being_built)) >= self.config.train_batch_size:
                batch_to_add = concat_samples(self.batch_being_built)
                self.batches_to_place_on_learner.append(batch_to_add)
                self.batch_being_built = []
        for batch in batches:
            self.batch_being_built.append(batch)
            aggregate_into_larger_batch()

    def get_samples_from_workers(self, return_object_refs: Optional[bool]=False) -> List[Tuple[int, Union[ObjectRef, SampleBatchType]]]:
        """Get samples from rollout workers for training.

        Args:
            return_object_refs: If True, return ObjectRefs instead of the samples
                directly. This is useful when using aggregator workers so that data
                collected on rollout workers is directly de referenced on the aggregator
                workers instead of first in the driver and then on the aggregator
                workers.

        Returns:
            a list of tuples of (worker_index, sample batch or ObjectRef to a sample
                batch)

        """
        with self._timers[SAMPLE_TIMER]:
            if self.workers.num_healthy_remote_workers() > 0:
                self.workers.foreach_worker_async(lambda worker: worker.sample(), healthy_only=True)
                sample_batches: List[Tuple[int, ObjectRef]] = self.workers.fetch_ready_async_reqs(timeout_seconds=self._timeout_s_sampler_manager, return_obj_refs=return_object_refs)
            elif self.workers.local_worker() and self.workers.local_worker().async_env is not None:
                sample_batch = self.workers.local_worker().sample()
                if return_object_refs:
                    sample_batch = ray.put(sample_batch)
                sample_batches = [(0, sample_batch)]
            else:
                return []
        return sample_batches

    def learn_on_processed_samples(self) -> ResultDict:
        """Update the learner group with the latest batch of processed samples.

        Returns:
            Aggregated results from the learner group after an update is completed.

        """
        if self.batches_to_place_on_learner:
            batches = self.batches_to_place_on_learner[:]
            self.batches_to_place_on_learner.clear()
            blocking = self.config.num_learner_workers == 0
            results = []
            for batch in batches:
                if blocking:
                    result = self.learner_group.update(batch, reduce_fn=_reduce_impala_results, num_iters=self.config.num_sgd_iter, minibatch_size=self.config.minibatch_size)
                    results = [result]
                else:
                    results = self.learner_group.async_update(batch, reduce_fn=_reduce_impala_results, num_iters=self.config.num_sgd_iter, minibatch_size=self.config.minibatch_size)
                for r in results:
                    self._counters[NUM_ENV_STEPS_TRAINED] += r[ALL_MODULES].pop(NUM_ENV_STEPS_TRAINED)
                    self._counters[NUM_AGENT_STEPS_TRAINED] += r[ALL_MODULES].pop(NUM_AGENT_STEPS_TRAINED)
            self._counters.update(self.learner_group.get_in_queue_stats())
            if results:
                return tree.map_structure(lambda *x: np.mean(x), *results)
        return {}

    def place_processed_samples_on_learner_thread_queue(self) -> None:
        """Place processed samples on the learner queue for training.

        NOTE: This method is called if self.config._enable_new_api_stack is False.

        """
        while self.batches_to_place_on_learner:
            batch = self.batches_to_place_on_learner[0]
            try:
                self._learner_thread.inqueue.put(batch, block=True)
                self.batches_to_place_on_learner.pop(0)
                self._counters['num_samples_added_to_queue'] += batch.agent_steps() if self.config.count_steps_by == 'agent_steps' else batch.count
            except queue.Full:
                self._counters['num_times_learner_queue_full'] += 1

    def process_trained_results(self) -> ResultDict:
        """Process training results that are outputed by the learner thread.

        NOTE: This method is called if self.config._enable_new_api_stack is False.

        Returns:
            Aggregated results from the learner thread after an update is completed.

        """
        num_env_steps_trained = 0
        num_agent_steps_trained = 0
        learner_infos = []
        for _ in range(self._learner_thread.outqueue.qsize()):
            env_steps, agent_steps, learner_results = self._learner_thread.outqueue.get(timeout=0.001)
            num_env_steps_trained += env_steps
            num_agent_steps_trained += agent_steps
            if learner_results:
                learner_infos.append(learner_results)
        if not learner_infos:
            final_learner_info = copy.deepcopy(self._learner_thread.learner_info)
        else:
            builder = LearnerInfoBuilder()
            for info in learner_infos:
                builder.add_learn_on_batch_results_multi_agent(info)
            final_learner_info = builder.finalize()
        self._counters[NUM_ENV_STEPS_TRAINED] += num_env_steps_trained
        self._counters[NUM_AGENT_STEPS_TRAINED] += num_agent_steps_trained
        return final_learner_info

    def process_experiences_directly(self, worker_to_sample_batches: List[Tuple[int, SampleBatch]]) -> List[SampleBatchType]:
        """Process sample batches directly on the driver, for training.

        Args:
            worker_to_sample_batches: List of (worker_id, sample_batch) tuples.

        Returns:
            Batches that have been processed by the mixin buffer.

        """
        batches = [b for _, b in worker_to_sample_batches]
        processed_batches = []
        for batch in batches:
            assert not isinstance(batch, ObjectRef), 'process_experiences_directly can not handle ObjectRefs. '
            batch = batch.decompress_if_needed()
            self.local_mixin_buffer.add(batch)
            batch = self.local_mixin_buffer.replay(_ALL_POLICIES)
            if batch:
                processed_batches.append(batch)
        return processed_batches

    def process_experiences_tree_aggregation(self, worker_to_sample_batches_refs: List[Tuple[int, ObjectRef]]) -> List[SampleBatchType]:
        """Process sample batches using tree aggregation workers.

        Args:
            worker_to_sample_batches_refs: List of (worker_id, sample_batch_ref)

        NOTE: This will provide speedup when sample batches have been compressed,
        and the decompression can happen on the aggregation workers in parallel to
        the training.

        Returns:
            Batches that have been processed by the mixin buffers on the aggregation
            workers.

        """

        def _process_episodes(actor, batch):
            return actor.process_episodes(ray.get(batch))
        for _, batch in worker_to_sample_batches_refs:
            assert isinstance(batch, ObjectRef), f'For efficiency, process_experiences_tree_aggregation should be given ObjectRefs instead of {type(batch)}.'
            aggregator_id = random.choice(self._aggregator_actor_manager.healthy_actor_ids())
            calls_placed = self._aggregator_actor_manager.foreach_actor_async(partial(_process_episodes, batch=batch), remote_actor_ids=[aggregator_id])
            if calls_placed <= 0:
                self._counters['num_times_no_aggregation_worker_available'] += 1
        waiting_processed_sample_batches: RemoteCallResults = self._aggregator_actor_manager.fetch_ready_async_reqs(timeout_seconds=self._timeout_s_aggregator_manager)
        handle_remote_call_result_errors(waiting_processed_sample_batches, self.config.ignore_worker_failures)
        return [b.get() for b in waiting_processed_sample_batches.ignore_errors()]

    def update_workers_from_learner_group(self, workers_that_need_updates: Set[int], policy_ids: Optional[List[PolicyID]]=None):
        """Updates all RolloutWorkers that require updating.

        Updates only if NUM_TRAINING_STEP_CALLS_SINCE_LAST_SYNCH_WORKER_WEIGHTS has been
        reached and the worker has sent samples in this iteration. Also only updates
        those policies, whose IDs are given via `policies` (if None, update all
        policies).

        Args:
            workers_that_need_updates: Set of worker IDs that need to be updated.
            policy_ids: Optional list of Policy IDs to update. If None, will update all
                policies on the to-be-updated workers.
        """
        self._counters[NUM_TRAINING_STEP_CALLS_SINCE_LAST_SYNCH_WORKER_WEIGHTS] += 1
        if self._counters[NUM_TRAINING_STEP_CALLS_SINCE_LAST_SYNCH_WORKER_WEIGHTS] >= self.config.broadcast_interval and workers_that_need_updates:
            self._counters[NUM_TRAINING_STEP_CALLS_SINCE_LAST_SYNCH_WORKER_WEIGHTS] = 0
            self._counters[NUM_SYNCH_WORKER_WEIGHTS] += 1
            weights = self.learner_group.get_weights(policy_ids)
            if self.config.num_rollout_workers == 0:
                worker = self.workers.local_worker()
                worker.set_weights(weights)
            else:
                weights_ref = ray.put(weights)
                self.workers.foreach_worker(func=lambda w: w.set_weights(ray.get(weights_ref)), local_worker=False, remote_worker_ids=list(workers_that_need_updates), timeout_seconds=0)
                if self.config.create_env_on_local_worker:
                    self.workers.local_worker().set_weights(weights)

    def update_workers_if_necessary(self, workers_that_need_updates: Set[int], policy_ids: Optional[List[PolicyID]]=None) -> None:
        """Updates all RolloutWorkers that require updating.

        Updates only if NUM_TRAINING_STEP_CALLS_SINCE_LAST_SYNCH_WORKER_WEIGHTS has been
        reached and the worker has sent samples in this iteration. Also only updates
        those policies, whose IDs are given via `policies` (if None, update all
        policies).

        Args:
            workers_that_need_updates: Set of worker IDs that need to be updated.
            policy_ids: Optional list of Policy IDs to update. If None, will update all
                policies on the to-be-updated workers.
        """
        local_worker = self.workers.local_worker()
        if self.config.policy_states_are_swappable:
            local_worker.lock()
        global_vars = {'timestep': self._counters[NUM_AGENT_STEPS_TRAINED], 'num_grad_updates_per_policy': {pid: local_worker.policy_map[pid].num_grad_updates for pid in policy_ids or []}}
        local_worker.set_global_vars(global_vars, policy_ids=policy_ids)
        if self.config.policy_states_are_swappable:
            local_worker.unlock()
        self._counters[NUM_TRAINING_STEP_CALLS_SINCE_LAST_SYNCH_WORKER_WEIGHTS] += 1
        if self.workers.num_remote_workers() > 0 and self._counters[NUM_TRAINING_STEP_CALLS_SINCE_LAST_SYNCH_WORKER_WEIGHTS] >= self.config.broadcast_interval and workers_that_need_updates:
            if self.config.policy_states_are_swappable:
                local_worker.lock()
            weights = local_worker.get_weights(policy_ids)
            if self.config.policy_states_are_swappable:
                local_worker.unlock()
            weights = ray.put(weights)
            self._learner_thread.policy_ids_updated.clear()
            self._counters[NUM_TRAINING_STEP_CALLS_SINCE_LAST_SYNCH_WORKER_WEIGHTS] = 0
            self._counters[NUM_SYNCH_WORKER_WEIGHTS] += 1
            self.workers.foreach_worker(func=lambda w: w.set_weights(ray.get(weights), global_vars), local_worker=False, remote_worker_ids=list(workers_that_need_updates), timeout_seconds=0)

    def _get_additional_update_kwargs(self, train_results: dict) -> dict:
        """Returns the kwargs to `LearnerGroup.additional_update()`.

        Should be overridden by subclasses to specify wanted/needed kwargs for
        their own implementation of `Learner.additional_update_for_module()`.
        """
        return {}

    @override(Algorithm)
    def _compile_iteration_results(self, *args, **kwargs):
        result = super()._compile_iteration_results(*args, **kwargs)
        if not self.config._enable_new_api_stack:
            result = self._learner_thread.add_learner_metrics(result, overwrite_learner_info=False)
        return result