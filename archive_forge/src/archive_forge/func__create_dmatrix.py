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
def _create_dmatrix(feature_names: Optional[FeatureNames], feature_types: Optional[Union[Any, List[Any]]], feature_weights: Optional[Any], missing: float, nthread: int, enable_categorical: bool, parts: Optional[_DataParts]) -> DMatrix:
    """Get data that local to worker from DaskDMatrix.

    Returns
    -------
    A DMatrix object.

    """
    worker = distributed.get_worker()
    list_of_parts = parts
    if list_of_parts is None:
        msg = f'worker {worker.address} has an empty DMatrix.'
        LOGGER.warning(msg)
        d = DMatrix(numpy.empty((0, 0)), feature_names=feature_names, feature_types=feature_types, enable_categorical=enable_categorical)
        return d
    T = TypeVar('T')

    def concat_or_none(data: Sequence[Optional[T]]) -> Optional[T]:
        if any((part is None for part in data)):
            return None
        return dconcat(data)
    unzipped_dict = _get_worker_parts(list_of_parts)
    concated_dict: Dict[str, Any] = {}
    for key, value in unzipped_dict.items():
        v = concat_or_none(value)
        concated_dict[key] = v
    dmatrix = DMatrix(**concated_dict, missing=missing, feature_names=feature_names, feature_types=feature_types, nthread=nthread, enable_categorical=enable_categorical, feature_weights=feature_weights)
    return dmatrix