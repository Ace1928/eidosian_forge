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
class EFstrType(Enum):
    """Calculate score for every feature by values change."""
    PredictionValuesChange = 0
    'Calculate score for every feature by loss change'
    LossFunctionChange = 1
    'Use LossFunctionChange for ranking models and PredictionValuesChange otherwise'
    FeatureImportance = 2
    'Calculate pairwise score between every feature.'
    Interaction = 3
    'Calculate SHAP Values for every object.'
    ShapValues = 4
    'Calculate most important features explaining difference in predictions for a pair of documents'
    PredictionDiff = 5
    'Calculate SHAP Interaction Values pairwise between every feature for every object.'
    ShapInteractionValues = 6
    'Calculate SAGE Values for every feature'
    SageValues = 7