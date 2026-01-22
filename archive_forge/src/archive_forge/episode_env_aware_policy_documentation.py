import numpy as np
import tree
from gymnasium.spaces import Box
from ray.rllib.core.models.base import STATE_OUT
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.examples.rl_module.episode_env_aware_rlm import StatefulRandomRLModule
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override
from ray.rllib.utils.numpy import convert_to_numpy
Dummy method for compatibility with TorchRLModule.

                This is hit when RolloutWorker tries to compile TorchRLModule.