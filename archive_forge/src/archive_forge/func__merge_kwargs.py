from typing import Tuple
import gymnasium as gym
import abc
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.utils.typing import TensorType, Union
from ray.rllib.utils.annotations import override
@staticmethod
def _merge_kwargs(**kwargs):
    """Checks if keys in kwargs don't clash with partial_kwargs."""
    overlap = set(kwargs) & set(partial_kwargs)
    if overlap:
        raise ValueError(f'Cannot override the following kwargs: {overlap}.\nThis is because they were already set at the time this partial class was defined.')
    merged_kwargs = {**partial_kwargs, **kwargs}
    return merged_kwargs