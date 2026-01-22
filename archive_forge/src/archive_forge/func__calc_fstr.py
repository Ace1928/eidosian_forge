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
def _calc_fstr(self, type, pool, reference_data, thread_count, verbose, model_output, shap_mode, interaction_indices, shap_calc_type, sage_n_samples, sage_batch_size, sage_detect_convergence):
    return self._object._calc_fstr(type.name, pool, reference_data, thread_count, verbose, model_output, shap_mode, interaction_indices, shap_calc_type, sage_n_samples, sage_batch_size, sage_detect_convergence)