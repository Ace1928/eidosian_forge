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
def _get_features_indices(features, feature_names):
    """
        Parameters
        ----------
        features :
            must be a sequence of either integers or strings
            if it contains strings 'feature_names' parameter must be defined and string ids from 'features'
            must represent a subset of in 'feature_names'

        feature_names :
            A sequence of string ids for features or None.
            Used to get feature indices for string ids in 'features' parameter
    """
    if not isinstance(features, (Sequence, np.ndarray)) or isinstance(features, (str, bytes, bytearray)):
        raise CatBoostError('feature names should be a sequence, but got ' + repr(features))
    if feature_names is not None:
        return [feature_names.index(f) if isinstance(f, STRING_TYPES) else f for f in features]
    else:
        for f in features:
            if isinstance(f, STRING_TYPES):
                raise CatBoostError("features parameter contains string value '{}' but feature names for a dataset are not specified".format(f))
    return features