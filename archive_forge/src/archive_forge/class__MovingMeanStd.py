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
class _MovingMeanStd:
    """Track moving mean, std and count."""

    def __init__(self, epsilon: float=0.0001, shape: Optional[List[int]]=None):
        """Initialize object.

        Args:
            epsilon: Initial count.
            shape: Shape of the trackables mean and std.
        """
        if not shape:
            shape = []
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Normalize input batch using moving mean and std.

        Args:
            inputs: Input batch to normalize.

        Returns:
            Logarithmic scaled normalized output.
        """
        batch_mean = np.mean(inputs, axis=0)
        batch_var = np.var(inputs, axis=0)
        batch_count = inputs.shape[0]
        self.update_params(batch_mean, batch_var, batch_count)
        return np.log(inputs / self.std + 1)

    def update_params(self, batch_mean: float, batch_var: float, batch_count: float) -> None:
        """Update moving mean, std and count.

        Args:
            batch_mean: Input batch mean.
            batch_var: Input batch variance.
            batch_count: Number of cases in the batch.
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta + batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.power(delta, 2) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    @property
    def std(self) -> float:
        """Get moving standard deviation.

        Returns:
            Returns moving standard deviation.
        """
        return np.sqrt(self.var)