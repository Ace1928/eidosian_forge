import gymnasium as gym
import numpy as np
from typing import Optional, List, Mapping, Iterable, Dict
import tree
import abc
from ray.rllib.models.distributions import Distribution
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType, Union, Tuple
def to_deterministic(self) -> 'TorchMultiDistribution':
    flat_deterministic_dists = [dist.to_deterministic() for dist in self._flat_child_distributions]
    deterministic_dists = tree.unflatten_as(self._original_struct, flat_deterministic_dists)
    return TorchMultiDistribution(deterministic_dists)