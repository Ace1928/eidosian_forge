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
def _plot_feature_statistics_units(statistics, pool_names, feature_name, max_cat_features_on_plot):
    if 'cat_values' in statistics[0].keys() and len(statistics[0]['cat_values']) > max_cat_features_on_plot:
        figs = []
        for begin in range(0, len(statistics[0]['cat_values']), max_cat_features_on_plot):
            end = begin + max_cat_features_on_plot
            statistics_keys = ['cat_values', 'mean_target', 'mean_weighted_target', 'mean_prediction', 'objects_per_bin', 'predictions_on_varying_feature']
            sub_statistics = dict([(k, dict([(key, stats[key][begin:end]) for key in statistics_keys])) for k, stats in statistics])
            fig = _build_binarized_feature_statistics_fig(sub_statistics, pool_names)
            feature_name_with_part_suffix = '{}_parts[{}:{}]'.format(feature_name, begin, end)
            figs += [(fig, feature_name_with_part_suffix)]
        return figs
    else:
        fig = _build_binarized_feature_statistics_fig(statistics, pool_names)
        return [(fig, feature_name)]