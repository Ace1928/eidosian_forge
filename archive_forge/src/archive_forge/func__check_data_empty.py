from contextlib import contextmanager  # noqa E402
from copy import deepcopy
import logging
import sys
import os
from collections import OrderedDict, defaultdict
from six import iteritems, string_types, integer_types
import warnings
import numpy as np
import ctypes
import platform
import tempfile
import shutil
import json
from enum import Enum
from operator import itemgetter
import threading
import scipy.sparse
from .plot_helpers import save_plot_file, try_plot_offline, OfflineMetricVisualizer
from . import _catboost
from .metrics import BuiltinMetric
def _check_data_empty(self, data):
    """
        Check that data is not empty (0 objects is ok).
        note: already checked if data is FeatureType, so no need to check again
        """
    if isinstance(data, PATH_TYPES):
        if not data:
            raise CatBoostError('Features filename is empty.')
    elif isinstance(data, (ARRAY_TYPES, SPARSE_MATRIX_TYPES)):
        if isinstance(data, list):
            data_shape = np.shape(np.asarray(data, dtype=object))
        else:
            data_shape = np.shape(data)
        if len(data_shape) == 1 and data_shape[0] > 0:
            if isinstance(data[0], Iterable):
                data_shape = tuple(data_shape + tuple([len(data[0])]))
            else:
                data_shape = tuple(data_shape + tuple([1]))
        if not len(data_shape) == 2:
            raise CatBoostError('Input data has invalid shape: {}. Must be 2 dimensional'.format(data_shape))
        if data_shape[1] == 0:
            raise CatBoostError('Input data must have at least one feature')