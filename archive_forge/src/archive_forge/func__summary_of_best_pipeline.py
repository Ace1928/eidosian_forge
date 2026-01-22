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
def _summary_of_best_pipeline(self, features, target):
    """Print out best pipeline at the end of optimization process.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix

        target: array-like {n_samples}
            List of class labels for prediction

        Returns
        -------
        self: object
            Returns a copy of the fitted TPOT object
        """
    if not self._optimized_pipeline:
        raise RuntimeError('There was an error in the TPOT optimization process. This could be because the data was not formatted properly (e.g. nan values became a third class), or because data for a regression problem was provided to the TPOTClassifier object. Please make sure you passed the data to TPOT correctly.')
    else:
        self.fitted_pipeline_ = self._toolbox.compile(expr=self._optimized_pipeline)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.fitted_pipeline_.fit(features, target)
        if self.verbosity in [1, 2]:
            if self.verbosity >= 2:
                print('')
            optimized_pipeline_str = self.clean_pipeline_string(self._optimized_pipeline)
            print('Best pipeline:', optimized_pipeline_str)
        self.pareto_front_fitted_pipelines_ = {}
        for pipeline in self._pareto_front.items:
            self.pareto_front_fitted_pipelines_[str(pipeline)] = self._toolbox.compile(expr=pipeline)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.pareto_front_fitted_pipelines_[str(pipeline)].fit(features, target)