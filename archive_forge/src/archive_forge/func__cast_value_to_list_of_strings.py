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
def _cast_value_to_list_of_strings(params, key):
    if key in params:
        if isinstance(params[key], STRING_TYPES):
            params[key] = [params[key]]
        if not isinstance(params[key], Sequence):
            raise CatBoostError('Invalid `' + key + '` type={} : must be string or list of strings.'.format(type(params[key])))