import gymnasium as gym
import tree
import numpy as np
from typing import Optional, List, Mapping, Iterable, Dict
import abc
from ray.rllib.models.distributions import Distribution
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
from ray.rllib.utils.typing import TensorType, Union, Tuple
@DeveloperAPI
class TfDistribution(Distribution, abc.ABC):
    """Wrapper class for tfp.distributions."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._dist = self._get_tf_distribution(*args, **kwargs)

    @abc.abstractmethod
    def _get_tf_distribution(self, *args, **kwargs) -> 'tfp.distributions.Distribution':
        """Returns the tfp.distributions.Distribution object to use."""

    @override(Distribution)
    def logp(self, value: TensorType, **kwargs) -> TensorType:
        return self._dist.log_prob(value, **kwargs)

    @override(Distribution)
    def entropy(self) -> TensorType:
        return self._dist.entropy()

    @override(Distribution)
    def kl(self, other: 'Distribution') -> TensorType:
        return self._dist.kl_divergence(other._dist)

    @override(Distribution)
    def sample(self, *, sample_shape=()) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        sample = self._dist.sample(sample_shape)
        return sample

    @override(Distribution)
    def rsample(self, *, sample_shape=()) -> Union[TensorType, Tuple[TensorType, TensorType]]:
        raise NotImplementedError