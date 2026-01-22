import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree
import random
from typing import Union, Optional
from ray.rllib.models.torch.torch_action_dist import TorchMultiActionDistribution
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration, TensorType
from ray.rllib.utils.framework import try_import_tf, try_import_torch, get_variable
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.schedules import Schedule, PiecewiseSchedule
from ray.rllib.utils.torch_utils import FLOAT_MIN
Torch method to produce an epsilon exploration action.

        Args:
            action_distribution: The instantiated
                ActionDistribution object to work with when creating
                exploration actions.

        Returns:
            The exploration-action.
        