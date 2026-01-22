import collections
import logging
import platform
import socket
import warnings
from collections import defaultdict
from contextlib import contextmanager
from functools import partial, update_wrapper
from threading import Thread
from typing import (
import numpy
from . import collective, config
from ._typing import _T, FeatureNames, FeatureTypes, ModelIn
from .callback import TrainingCallback
from .compat import DataFrame, LazyLoader, concat, lazy_isinstance
from .core import (
from .data import _is_cudf_ser, _is_cupy_array
from .sklearn import (
from .tracker import RabitTracker, get_host_ip
from .training import train as worker_train
class DaskPartitionIter(DataIter):
    """A data iterator for `DaskQuantileDMatrix`."""

    def __init__(self, data: List[Any], label: Optional[List[Any]]=None, weight: Optional[List[Any]]=None, base_margin: Optional[List[Any]]=None, qid: Optional[List[Any]]=None, label_lower_bound: Optional[List[Any]]=None, label_upper_bound: Optional[List[Any]]=None, feature_names: Optional[FeatureNames]=None, feature_types: Optional[Union[Any, List[Any]]]=None, feature_weights: Optional[Any]=None) -> None:
        self._data = data
        self._label = label
        self._weight = weight
        self._base_margin = base_margin
        self._qid = qid
        self._label_lower_bound = label_lower_bound
        self._label_upper_bound = label_upper_bound
        self._feature_names = feature_names
        self._feature_types = feature_types
        self._feature_weights = feature_weights
        assert isinstance(self._data, collections.abc.Sequence)
        types = (collections.abc.Sequence, type(None))
        assert isinstance(self._label, types)
        assert isinstance(self._weight, types)
        assert isinstance(self._base_margin, types)
        assert isinstance(self._label_lower_bound, types)
        assert isinstance(self._label_upper_bound, types)
        self._iter = 0
        super().__init__()

    def _get(self, attr: str) -> Optional[Any]:
        if getattr(self, attr) is not None:
            return getattr(self, attr)[self._iter]
        return None

    def data(self) -> Any:
        """Utility function for obtaining current batch of data."""
        return self._data[self._iter]

    def reset(self) -> None:
        """Reset the iterator"""
        self._iter = 0

    def next(self, input_data: Callable) -> int:
        """Yield next batch of data"""
        if self._iter == len(self._data):
            return 0
        input_data(data=self.data(), label=self._get('_label'), weight=self._get('_weight'), group=None, qid=self._get('_qid'), base_margin=self._get('_base_margin'), label_lower_bound=self._get('_label_lower_bound'), label_upper_bound=self._get('_label_upper_bound'), feature_names=self._feature_names, feature_types=self._feature_types, feature_weights=self._feature_weights)
        self._iter += 1
        return 1