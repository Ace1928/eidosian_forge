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
def _process_synonyms_group(synonyms, params):
    assert len(synonyms) > 1, 'there should be more than one synonym'
    value = None
    for synonym in synonyms:
        if synonym in params:
            if value is not None:
                raise CatBoostError('only one of the parameters ' + ', '.join(synonyms) + ' should be initialized.')
            value = params[synonym]
            del params[synonym]
    if value is not None:
        params[synonyms[0]] = value