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
def _virtual_ensembles_predict(self, data, prediction_type, ntree_end, virtual_ensembles_count, thread_count, verbose, parent_method_name):
    verbose = verbose or self.get_param('verbose')
    if verbose is None:
        verbose = False
    data, data_is_single_object = self._process_predict_input_data(data, parent_method_name, thread_count)
    self._validate_prediction_type(prediction_type, ['VirtEnsembles', 'TotalUncertainty'])
    if ntree_end == 0:
        ntree_end = self.tree_count_
    predictions = self._base_virtual_ensembles_predict(data, prediction_type, ntree_end, virtual_ensembles_count, thread_count, verbose)
    if prediction_type == 'VirtEnsembles':
        shape = predictions.shape
        predictions = predictions.reshape(shape[0], virtual_ensembles_count, int(shape[1] / virtual_ensembles_count))
    return predictions[0] if data_is_single_object else predictions