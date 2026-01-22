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
def _build_train_pool(X, y, cat_features, text_features, embedding_features, pairs, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline, column_description):
    train_pool = None
    if isinstance(X, Pool):
        train_pool = X
        if any((v is not None for v in [cat_features, text_features, embedding_features, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline])):
            raise CatBoostError('cat_features, text_features, embedding_features, sample_weight, group_id, group_weight, subgroup_id, pairs_weight, baseline should have the None type when X has catboost.Pool type.')
        if not X.has_label() and X.num_pairs() == 0:
            raise CatBoostError('Label in X has not been initialized.')
        if y is not None:
            raise CatBoostError('Incorrect value of y: X is catboost.Pool object, y must be initialized inside catboost.Pool.')
    elif isinstance(X, PATH_TYPES):
        train_pool = Pool(data=X, pairs=pairs, column_description=column_description)
    else:
        if y is None:
            raise CatBoostError('y has not initialized in fit(): X is not catboost.Pool object, y must be not None in fit().')
        train_pool = Pool(X, y, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features, pairs=pairs, weight=sample_weight, group_id=group_id, group_weight=group_weight, subgroup_id=subgroup_id, pairs_weight=pairs_weight, baseline=baseline)
    return train_pool