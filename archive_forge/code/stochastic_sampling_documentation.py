import functools
import gymnasium as gym
import numpy as np
from typing import Optional, Union
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.exploration.random import Random
from ray.rllib.utils.framework import (
from ray.rllib.utils.tf_utils import zero_logps_from_actions
Initializes a StochasticSampling Exploration object.

        Args:
            action_space: The gym action space used by the environment.
            framework: One of None, "tf", "torch".
            model: The ModelV2 used by the owning Policy.
            random_timesteps: The number of timesteps for which to act
                completely randomly. Only after this number of timesteps,
                actual samples will be drawn to get exploration actions.
        