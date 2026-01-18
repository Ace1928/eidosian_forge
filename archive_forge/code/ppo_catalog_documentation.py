import gymnasium as gym
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.configs import (
from ray.rllib.core.models.base import Encoder, ActorCriticEncoder, Model
from ray.rllib.utils import override
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic
Builds the value function head.

        The default behavior is to build the head from the vf_head_config.
        This can be overridden to build a custom value function head as a means of
        configuring the behavior of a PPORLModule implementation.

        Args:
            framework: The framework to use. Either "torch" or "tf2".

        Returns:
            The value function head.
        