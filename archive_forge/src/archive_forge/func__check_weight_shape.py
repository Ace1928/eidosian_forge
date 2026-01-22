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
def _check_weight_shape(self, weight, samples_count):
    """
        Check weight length.
        """
    if len(weight) != samples_count:
        raise CatBoostError('Length of weight={} and length of data={} are different.'.format(len(weight), samples_count))
    if not isinstance(weight[0], (INTEGER_TYPES, FLOAT_TYPES)):
        raise CatBoostError('Invalid weight value type={}: must be 1 dimensional data with int, float or long types.'.format(type(weight[0])))