import gymnasium as gym
from ray.rllib.algorithms.ppo.ppo_catalog import _check_if_diag_gaussian
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.core.models.configs import FreeLogStdMLPHeadConfig, MLPHeadConfig
from ray.rllib.core.models.base import Model
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic
Builds the policy head.

        The default behavior is to build the head from the pi_head_config.
        This can be overridden to build a custom policy head as a means of configuring
        the behavior of a BCRLModule implementation.

        Args:
            framework: The framework to use. Either "torch" or "tf2".

        Returns:
            The policy head.
        