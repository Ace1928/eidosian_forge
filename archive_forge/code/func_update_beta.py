from gymnasium.spaces import Box, Discrete, Space
import numpy as np
from typing import List, Optional, Union
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.tf_utils import get_placeholder
from ray.rllib.utils.typing import FromConfigSpec, ModelConfigDict, TensorType
@PublicAPI
def update_beta(beta_schedule: str, beta: float, rho: float, step: int) -> float:
    """Update beta based on schedule and training step.

    Args:
        beta_schedule: Schedule for beta update.
        beta: Initial beta.
        rho: Schedule decay parameter.
        step: Current training iteration.

    Returns:
        Updated beta as per input schedule.
    """
    if beta_schedule == 'linear_decay':
        return beta * (1.0 - rho) ** step
    return beta