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
def _add_operators(self):
    main_type = ['Classifier', 'Regressor', 'Selector', 'Transformer']
    ret_types = []
    self.op_list = []
    if self.template == None:
        step_in_type = np.ndarray
        step_ret_type = Output_Array
        for operator in self.operators:
            arg_types = operator.parameter_types()[0][1:]
            p_types = ([step_in_type] + arg_types, step_ret_type)
            if operator.root:
                self._pset.addPrimitive(operator, *p_types)
            tree_p_types = ([step_in_type] + arg_types, step_in_type)
            self._pset.addPrimitive(operator, *tree_p_types)
            self._import_hash_and_add_terminals(operator, arg_types)
        self._pset.addPrimitive(CombineDFs(), [step_in_type, step_in_type], step_in_type)
    else:
        gp_types = {}
        for idx, step in enumerate(self._template_comp):
            if idx:
                step_in_type = ret_types[-1]
            else:
                step_in_type = np.ndarray
            if step != 'CombineDFs':
                if idx < len(self._template_comp) - 1:
                    step_ret_type_name = 'Ret_{}'.format(idx)
                    step_ret_type = type(step_ret_type_name, (object,), {})
                    ret_types.append(step_ret_type)
                else:
                    step_ret_type = Output_Array
            check_template = True
            if step == 'CombineDFs':
                self._pset.addPrimitive(CombineDFs(), [step_in_type, step_in_type], step_in_type)
            elif main_type.count(step):
                ops = [op for op in self.operators if op.type() == step]
                for operator in ops:
                    arg_types = operator.parameter_types()[0][1:]
                    p_types = ([step_in_type] + arg_types, step_ret_type)
                    self._pset.addPrimitive(operator, *p_types)
                    self._import_hash_and_add_terminals(operator, arg_types)
            else:
                try:
                    operator = next((op for op in self.operators if op.__name__ == step))
                except:
                    raise ValueError('An error occured while attempting to read the specified template. Please check a step named {}'.format(step))
                arg_types = operator.parameter_types()[0][1:]
                p_types = ([step_in_type] + arg_types, step_ret_type)
                self._pset.addPrimitive(operator, *p_types)
                self._import_hash_and_add_terminals(operator, arg_types)
    self.ret_types = [np.ndarray, Output_Array] + ret_types