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
def _fit_init(self):
    if not self.warm_start or not hasattr(self, '_pareto_front'):
        self._pop = []
        self._pareto_front = None
        self._last_optimized_pareto_front = None
        self._last_optimized_pareto_front_n_gens = 0
        self._setup_config(self.config_dict)
        self._setup_template(self.template)
        self.operators = []
        self.arguments = []
        make_pipeline_func = self._get_make_pipeline_func()
        for key in sorted(self._config_dict.keys()):
            op_class, arg_types = TPOTOperatorClassFactory(key, self._config_dict[key], BaseClass=Operator, ArgBaseClass=ARGType, verbose=self.verbosity)
            if op_class:
                self.operators.append(op_class)
                self.arguments += arg_types
        self.operators_context = {'make_pipeline': make_pipeline_func, 'make_union': make_union, 'StackingEstimator': StackingEstimator, 'FunctionTransformer': FunctionTransformer, 'copy': copy}
        self._setup_pset()
        self._setup_toolbox()
        self.evaluated_individuals_ = {}
    self._optimized_pipeline = None
    self._optimized_pipeline_score = None
    self._exported_pipeline_text = []
    self.fitted_pipeline_ = None
    self._fitted_imputer = None
    self._imputed = False
    self._memory = None
    self._output_best_pipeline_period_seconds = 30
    self._max_mut_loops = 50
    if self.max_time_mins is None and self.generations is None:
        raise ValueError('Either the parameter generations should bet set or a maximum evaluation time should be defined via max_time_mins')
    if self.max_time_mins is not None and self.generations is None:
        self.generations = 1000000
    if not self.disable_update_check:
        update_check('tpot', __version__)
    if self.mutation_rate + self.crossover_rate > 1:
        raise ValueError('The sum of the crossover and mutation probabilities must be <= 1.0.')
    self._pbar = None
    if not self.log_file:
        self.log_file_ = sys.stdout
    elif isinstance(self.log_file, str):
        self.log_file_ = open(self.log_file, 'w')
    else:
        self.log_file_ = self.log_file
    self._setup_scoring_function(self.scoring)
    if self.subsample <= 0.0 or self.subsample > 1.0:
        raise ValueError('The subsample ratio of the training instance must be in the range (0.0, 1.0].')
    if self.n_jobs == 0:
        raise ValueError('The value 0 of n_jobs is invalid.')
    elif self.n_jobs < 0:
        self._n_jobs = cpu_count() + 1 + self.n_jobs
    else:
        self._n_jobs = self.n_jobs