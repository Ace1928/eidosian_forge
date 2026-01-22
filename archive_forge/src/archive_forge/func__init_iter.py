import os
import time
import logging
import warnings
from collections import namedtuple
import numpy as np
from . import io
from . import ndarray as nd
from . import symbol as sym
from . import optimizer as opt
from . import metric
from . import kvstore as kvs
from .context import Context, cpu
from .initializer import Uniform
from .optimizer import get_updater
from .executor_manager import DataParallelExecutorManager, _check_arguments, _load_data
from .io import DataDesc
from .base import mx_real_t
from .callback import LogValidationMetricsCallback # pylint: disable=wrong-import-position
def _init_iter(self, X, y, is_train):
    """Initialize the iterator given input."""
    if isinstance(X, (np.ndarray, nd.NDArray)):
        if y is None:
            if is_train:
                raise ValueError('y must be specified when X is numpy.ndarray')
            y = np.zeros(X.shape[0])
        if not isinstance(y, (np.ndarray, nd.NDArray)):
            raise TypeError('y must be ndarray when X is numpy.ndarray')
        if X.shape[0] != y.shape[0]:
            raise ValueError('The numbers of data points and labels not equal')
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.flatten()
        if y.ndim != 1:
            raise ValueError('Label must be 1D or 2D (with 2nd dimension being 1)')
        if is_train:
            return io.NDArrayIter(X, y, min(X.shape[0], self.numpy_batch_size), shuffle=is_train, last_batch_handle='roll_over')
        else:
            return io.NDArrayIter(X, y, min(X.shape[0], self.numpy_batch_size), shuffle=False)
    if not isinstance(X, io.DataIter):
        raise TypeError('X must be DataIter, NDArray or numpy.ndarray')
    return X