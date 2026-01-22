import gymnasium as gym
import numpy as np
from typing import Optional, List, Mapping, Iterable, Dict
import tree
import abc
from ray.rllib.models.distributions import Distribution
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType, Union, Tuple
@DeveloperAPI
class TorchDistribution(Distribution, abc.ABC):
    """Wrapper class for torch.distributions."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._dist = self._get_torch_distribution(*args, **kwargs)

    @abc.abstractmethod
    def _get_torch_distribution(self, *args, **kwargs) -> 'torch.distributions.Distribution':
        """Returns the torch.distributions.Distribution object to use."""

    @override(Distribution)
    def logp(self, value: TensorType, **kwargs) -> TensorType:
        return self._dist.log_prob(value, **kwargs)

    @override(Distribution)
    def entropy(self) -> TensorType:
        return self._dist.entropy()

    @override(Distribution)
    def kl(self, other: 'Distribution') -> TensorType:
        return torch.distributions.kl.kl_divergence(self._dist, other._dist)

    @override(Distribution)
    def sample(self, *, sample_shape=torch.Size()) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        sample = self._dist.sample(sample_shape)
        return sample

    @override(Distribution)
    def rsample(self, *, sample_shape=torch.Size()) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        rsample = self._dist.rsample(sample_shape)
        return rsample