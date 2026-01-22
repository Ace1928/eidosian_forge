from collections import defaultdict
from functools import partial
from typing import List, Tuple
import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.wrappers.atari_wrappers import NoopResetEnv, MaxAndSkipEnv
from ray.rllib.env.wrappers.dm_control_wrapper import DMCEnv
from ray.rllib.env.utils import _gym_env_creator
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.numpy import one_hot
from ray.tune.registry import ENV_CREATOR, _global_registry
class NormalizedImageEnv(gym.ObservationWrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=self.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return observation.astype(np.float32) / 128.0 - 1.0