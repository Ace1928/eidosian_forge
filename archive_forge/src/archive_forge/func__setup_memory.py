from __future__ import print_function
import random
import inspect
import warnings
import sys
from functools import partial
from datetime import datetime
from multiprocessing import cpu_count
import os
import re
import errno
from tempfile import mkdtemp
from shutil import rmtree
import types
import numpy as np
from pandas import DataFrame
from scipy import sparse
import deap
from deap import base, creator, tools, gp
from copy import copy, deepcopy
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y, check_consistent_length, check_array
from sklearn.pipeline import make_union, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import check_cv
from sklearn.utils.metaestimators import available_if
from joblib import Parallel, delayed, Memory
from update_checker import update_check
from ._version import __version__
from .operator_utils import TPOTOperatorClassFactory, Operator, ARGType
from .export_utils import (
from .decorators import _pre_test
from .builtins import CombineDFs, StackingEstimator
from .config.classifier_light import classifier_config_dict_light
from .config.regressor_light import regressor_config_dict_light
from .config.classifier_mdr import tpot_mdr_classifier_config_dict
from .config.regressor_mdr import tpot_mdr_regressor_config_dict
from .config.regressor_sparse import regressor_config_sparse
from .config.classifier_sparse import classifier_config_sparse
from .config.classifier_nn import classifier_config_nn
from .config.classifier_cuml import classifier_config_cuml
from .config.regressor_cuml import regressor_config_cuml
from .metrics import SCORERS
from .gp_types import Output_Array
from .gp_deap import (
def _setup_memory(self):
    """Setup Memory object for memory caching.
        """
    if self.memory:
        if isinstance(self.memory, str):
            if self.memory == 'auto':
                self._cachedir = mkdtemp()
            else:
                if not os.path.isdir(self.memory):
                    try:
                        os.makedirs(self.memory)
                    except:
                        raise ValueError('Could not create directory for memory caching: {}'.format(self.memory))
                self._cachedir = self.memory
            self._memory = Memory(location=self._cachedir, verbose=0)
        elif isinstance(self.memory, Memory):
            self._memory = self.memory
        else:
            raise ValueError('Could not recognize Memory object for pipeline caching. Please provide an instance of joblib.Memory, a path to a directory on your system, or "auto".')