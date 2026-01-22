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
def _base_eval_metrics(self, pool, metrics_description, ntree_start, ntree_end, eval_period, thread_count, result_dir, tmp_dir):
    metrics_description_list = metrics_description if isinstance(metrics_description, list) else [metrics_description]
    return self._object._base_eval_metrics(pool, metrics_description_list, ntree_start, ntree_end, eval_period, thread_count, result_dir, tmp_dir)