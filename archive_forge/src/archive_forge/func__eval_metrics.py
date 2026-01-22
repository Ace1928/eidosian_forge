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
def _eval_metrics(self, data, metrics, ntree_start, ntree_end, eval_period, thread_count, res_dir, tmp_dir, plot, plot_file, log_cout=None, log_cerr=None):
    if not self.is_fitted():
        raise CatBoostError('There is no trained model to evaluate metrics on. Use fit() to train model. Then call this method.')
    if not isinstance(data, Pool):
        raise CatBoostError('Invalid data type={}, must be catboost.Pool.'.format(type(data)))
    if data.is_empty_:
        raise CatBoostError('Data is empty.')
    if not isinstance(metrics, ARRAY_TYPES) and (not isinstance(metrics, STRING_TYPES)) and (not isinstance(metrics, BuiltinMetric)):
        raise CatBoostError('Invalid metrics type={}, must be list(), str() or one of builtin catboost.metrics.* class instances.'.format(type(metrics)))
    if not all(map(lambda metric: isinstance(metric, string_types) or isinstance(metric, BuiltinMetric), metrics)):
        raise CatBoostError('Invalid metric type: must be string() or one of builtin catboost.metrics.* class instances.')
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp()
    if isinstance(metrics, STRING_TYPES) or isinstance(metrics, BuiltinMetric):
        metrics = [metrics]
    metrics = stringify_builtin_metrics_list(metrics)
    with log_fixup(log_cout, log_cerr), plot_wrapper(plot, plot_file, 'Eval metrics plot', [res_dir]):
        metrics_score, metric_names = self._base_eval_metrics(data, metrics, ntree_start, ntree_end, eval_period, thread_count, res_dir, tmp_dir)
    return dict(zip(metric_names, metrics_score))