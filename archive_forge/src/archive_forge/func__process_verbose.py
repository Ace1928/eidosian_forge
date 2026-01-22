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
def _process_verbose(metric_period=None, verbose=None, logging_level=None, verbose_eval=None, silent=None):
    _check_param_type(metric_period, 'metric_period', (int,))
    _check_param_type(verbose, 'verbose', (bool, int))
    _check_param_type(logging_level, 'logging_level', (string_types,))
    _check_param_type(verbose_eval, 'verbose_eval', (bool, int))
    _check_param_type(silent, 'silent', (bool,))
    params = locals()
    exclusive_params = ['verbose', 'logging_level', 'verbose_eval', 'silent']
    at_most_one = sum((params.get(exclusive) is not None for exclusive in exclusive_params))
    if at_most_one > 1:
        raise CatBoostError('Only one of parameters {} should be set'.format(exclusive_params))
    if verbose is None:
        if silent is not None:
            verbose = not silent
        elif verbose_eval is not None:
            verbose = verbose_eval
    if verbose is not None:
        verbose = int(verbose)
    return (metric_period, verbose, logging_level)