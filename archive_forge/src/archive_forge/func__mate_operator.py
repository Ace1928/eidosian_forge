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
@_pre_test
def _mate_operator(self, ind1, ind2):
    for _ in range(self._max_mut_loops):
        ind1_copy, ind2_copy = (self._toolbox.clone(ind1), self._toolbox.clone(ind2))
        offspring, offspring2 = cxOnePoint(ind1_copy, ind2_copy)
        if str(offspring) not in self.evaluated_individuals_:
            offspring.statistics['predecessor'] = (str(ind1), str(ind2))
            offspring.statistics['mutation_count'] = ind1.statistics['mutation_count'] + ind2.statistics['mutation_count']
            offspring.statistics['crossover_count'] = ind1.statistics['crossover_count'] + ind2.statistics['crossover_count'] + 1
            offspring.statistics['generation'] = 'INVALID'
            break
    return (offspring, offspring2, self.evaluated_individuals_)