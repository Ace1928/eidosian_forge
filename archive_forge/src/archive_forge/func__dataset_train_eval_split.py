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
def _dataset_train_eval_split(self, train_pool, params, save_eval_pool):
    """
        returns:
            train_pool, eval_pool
                eval_pool will be uninitialized if save_eval_pool is false
        """
    is_classification = getattr(self, '_estimator_type', None) == 'classifier' or _CatBoostBase._is_classification_objective(params.get('loss_function', 'RMSE'))
    return train_pool.train_eval_split(params.get('has_time', False), is_classification, params['eval_fraction'], save_eval_pool)