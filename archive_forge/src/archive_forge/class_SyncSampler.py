import logging
import queue
import time
from abc import ABCMeta, abstractmethod
from collections import defaultdict, namedtuple
from typing import (
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.env.base_env import ASYNC_RESET_RETURN, BaseEnv, convert_to_base_env
from ray.rllib.evaluation.collectors.sample_collector import SampleCollector
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector
from ray.rllib.evaluation.env_runner_v2 import (
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.offline import InputReader
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import SampleBatch, concat_samples
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.debug import summarize
from ray.rllib.utils.deprecation import deprecation_warning, DEPRECATED_VALUE
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.numpy import convert_to_numpy, make_action_immutable
from ray.rllib.utils.spaces.space_utils import clip_action, unbatch, unsquash_action
from ray.rllib.utils.typing import (
from ray.util.debug import log_once
@DeveloperAPI
class SyncSampler(SamplerInput):
    """Sync SamplerInput that collects experiences when `get_data()` is called."""

    def __init__(self, *, worker: 'RolloutWorker', env: BaseEnv, clip_rewards: Union[bool, float], rollout_fragment_length: int, count_steps_by: str='env_steps', callbacks: 'DefaultCallbacks', multiple_episodes_in_batch: bool=False, normalize_actions: bool=True, clip_actions: bool=False, observation_fn: Optional['ObservationFunction']=None, sample_collector_class: Optional[Type[SampleCollector]]=None, render: bool=False, policies=None, policy_mapping_fn=None, preprocessors=None, obs_filters=None, tf_sess=None, horizon=DEPRECATED_VALUE, soft_horizon=DEPRECATED_VALUE, no_done_at_end=DEPRECATED_VALUE):
        """Initializes a SyncSampler instance.

        Args:
            worker: The RolloutWorker that will use this Sampler for sampling.
            env: Any Env object. Will be converted into an RLlib BaseEnv.
            clip_rewards: True for +/-1.0 clipping,
                actual float value for +/- value clipping. False for no
                clipping.
            rollout_fragment_length: The length of a fragment to collect
                before building a SampleBatch from the data and resetting
                the SampleBatchBuilder object.
            count_steps_by: One of "env_steps" (default) or "agent_steps".
                Use "agent_steps", if you want rollout lengths to be counted
                by individual agent steps. In a multi-agent env,
                a single env_step contains one or more agent_steps, depending
                on how many agents are present at any given time in the
                ongoing episode.
            callbacks: The Callbacks object to use when episode
                events happen during rollout.
            multiple_episodes_in_batch: Whether to pack multiple
                episodes into each batch. This guarantees batches will be
                exactly `rollout_fragment_length` in size.
            normalize_actions: Whether to normalize actions to the
                action space's bounds.
            clip_actions: Whether to clip actions according to the
                given action_space's bounds.
            observation_fn: Optional multi-agent observation func to use for
                preprocessing observations.
            sample_collector_class: An optional SampleCollector sub-class to
                use to collect, store, and retrieve environment-, model-,
                and sampler data.
            render: Whether to try to render the environment after each step.
        """
        if log_once('deprecated_sync_sampler_args'):
            if policies is not None:
                deprecation_warning(old='policies')
            if policy_mapping_fn is not None:
                deprecation_warning(old='policy_mapping_fn')
            if preprocessors is not None:
                deprecation_warning(old='preprocessors')
            if obs_filters is not None:
                deprecation_warning(old='obs_filters')
            if tf_sess is not None:
                deprecation_warning(old='tf_sess')
            if horizon != DEPRECATED_VALUE:
                deprecation_warning(old='horizon', error=True)
            if soft_horizon != DEPRECATED_VALUE:
                deprecation_warning(old='soft_horizon', error=True)
            if no_done_at_end != DEPRECATED_VALUE:
                deprecation_warning(old='no_done_at_end', error=True)
        self.base_env = convert_to_base_env(env)
        self.rollout_fragment_length = rollout_fragment_length
        self.extra_batches = queue.Queue()
        self.perf_stats = _PerfStats(ema_coef=worker.config.sampler_perf_stats_ema_coef)
        if not sample_collector_class:
            sample_collector_class = SimpleListCollector
        self.sample_collector = sample_collector_class(worker.policy_map, clip_rewards, callbacks, multiple_episodes_in_batch, rollout_fragment_length, count_steps_by=count_steps_by)
        self.render = render
        if worker.config.enable_connectors:
            self._env_runner_obj = EnvRunnerV2(worker=worker, base_env=self.base_env, multiple_episodes_in_batch=multiple_episodes_in_batch, callbacks=callbacks, perf_stats=self.perf_stats, rollout_fragment_length=rollout_fragment_length, count_steps_by=count_steps_by, render=self.render)
            self._env_runner = self._env_runner_obj.run()
        else:
            self._env_runner = _env_runner(worker, self.base_env, self.extra_batches.put, normalize_actions, clip_actions, multiple_episodes_in_batch, callbacks, self.perf_stats, observation_fn, self.sample_collector, self.render)
        self.metrics_queue = queue.Queue()

    @override(SamplerInput)
    def get_data(self) -> SampleBatchType:
        while True:
            item = next(self._env_runner)
            if isinstance(item, RolloutMetrics):
                self.metrics_queue.put(item)
            else:
                return item

    @override(SamplerInput)
    def get_metrics(self) -> List[RolloutMetrics]:
        completed = []
        while True:
            try:
                completed.append(self.metrics_queue.get_nowait()._replace(perf_stats=self.perf_stats.get()))
            except queue.Empty:
                break
        return completed

    @override(SamplerInput)
    def get_extra_batches(self) -> List[SampleBatchType]:
        extra = []
        while True:
            try:
                extra.append(self.extra_batches.get_nowait())
            except queue.Empty:
                break
        return extra