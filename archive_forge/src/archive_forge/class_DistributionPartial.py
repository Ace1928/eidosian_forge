from typing import Tuple
import gymnasium as gym
import abc
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.utils.typing import TensorType, Union
from ray.rllib.utils.annotations import override
class DistributionPartial(parent_cls):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _merge_kwargs(**kwargs):
        """Checks if keys in kwargs don't clash with partial_kwargs."""
        overlap = set(kwargs) & set(partial_kwargs)
        if overlap:
            raise ValueError(f'Cannot override the following kwargs: {overlap}.\nThis is because they were already set at the time this partial class was defined.')
        merged_kwargs = {**partial_kwargs, **kwargs}
        return merged_kwargs

    @classmethod
    @override(parent_cls)
    def required_input_dim(cls, space: gym.Space, **kwargs) -> int:
        merged_kwargs = cls._merge_kwargs(**kwargs)
        assert space == merged_kwargs['space']
        return parent_cls.required_input_dim(**merged_kwargs)

    @classmethod
    @override(parent_cls)
    def from_logits(cls, logits: TensorType, **kwargs) -> 'DistributionPartial':
        merged_kwargs = cls._merge_kwargs(**kwargs)
        distribution = parent_cls.from_logits(logits, **merged_kwargs)
        distribution.__class__ = cls
        return distribution