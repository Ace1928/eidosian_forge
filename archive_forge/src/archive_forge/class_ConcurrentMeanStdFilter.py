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
class ConcurrentMeanStdFilter(MeanStdFilter):
    is_concurrent = True

    def __init__(self, *args, **kwargs):
        super(ConcurrentMeanStdFilter, self).__init__(*args, **kwargs)
        deprecation_warning(old='ConcurrentMeanStdFilter', error=False, help='ConcurrentMeanStd filters are only used for testing and will therefore be deprecated in the course of moving to the Connetors API, where testing of filters will be done by other means.')
        self._lock = threading.RLock()

        def lock_wrap(func):

            def wrapper(*args, **kwargs):
                with self._lock:
                    return func(*args, **kwargs)
            return wrapper
        self.__getattribute__ = lock_wrap(self.__getattribute__)

    def as_serializable(self) -> 'MeanStdFilter':
        """Returns non-concurrent version of current class"""
        other = MeanStdFilter(self.shape)
        other.sync(self)
        return other

    def copy(self) -> 'ConcurrentMeanStdFilter':
        """Returns a copy of Filter."""
        other = ConcurrentMeanStdFilter(self.shape)
        other.sync(self)
        return other

    def __repr__(self) -> str:
        return 'ConcurrentMeanStdFilter({}, {}, {}, {}, {}, {})'.format(self.shape, self.demean, self.destd, self.clip, self.running_stats, self.buffer)