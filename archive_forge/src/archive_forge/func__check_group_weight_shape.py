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
def _check_group_weight_shape(self, group_weight, samples_count):
    """
        Check group_weight length.
        """
    if len(group_weight) != samples_count:
        raise CatBoostError('Length of group_weight={} and length of data={} are different.'.format(len(group_weight), samples_count))
    if not isinstance(group_weight[0], FLOAT_TYPES):
        raise CatBoostError('Invalid group_weight value type={}: must be 1 dimensional data with float types.'.format(type(group_weight[0])))