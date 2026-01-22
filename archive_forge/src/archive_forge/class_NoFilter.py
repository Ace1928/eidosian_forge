import logging
import threading
import numpy as np
import tree  # pip install dm_tree
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.numpy import SMALL_NUMBER
from ray.rllib.utils.typing import TensorStructType
from ray.rllib.utils.serialization import _serialize_ndarray, _deserialize_ndarray
from ray.rllib.utils.deprecation import deprecation_warning
@DeveloperAPI
class NoFilter(Filter):
    is_concurrent = True

    def __call__(self, x: TensorStructType, update=True):
        if isinstance(x, (np.ndarray, dict, tuple)):
            return x
        try:
            return np.asarray(x)
        except Exception:
            raise ValueError('Failed to convert to array', x)

    def apply_changes(self, other: 'NoFilter', *args, **kwargs) -> None:
        pass

    def copy(self) -> 'NoFilter':
        return self

    def sync(self, other: 'NoFilter') -> None:
        pass

    def reset_buffer(self) -> None:
        pass

    def as_serializable(self) -> 'NoFilter':
        return self