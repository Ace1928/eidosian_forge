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
def _random_mutation_operator(self, individual, allow_shrink=True):
    """Perform a replacement, insertion, or shrink mutation on an individual.

        Parameters
        ----------
        individual: DEAP individual
            A list of pipeline operators and model parameters that can be
            compiled by DEAP into a callable function

        allow_shrink: bool (True)
            If True the `mutShrink` operator, which randomly shrinks the pipeline,
            is allowed to be chosen as one of the random mutation operators.
            If False, `mutShrink`  will never be chosen as a mutation operator.

        Returns
        -------
        mut_ind: DEAP individual
            Returns the individual with one of the mutations applied to it

        """
    if self.tree_structure:
        mutation_techniques = [partial(gp.mutInsert, pset=self._pset), partial(mutNodeReplacement, pset=self._pset)]
        number_of_primitives = sum((isinstance(node, deap.gp.Primitive) for node in individual))
        if number_of_primitives > 1 and allow_shrink:
            mutation_techniques.append(partial(gp.mutShrink))
    else:
        mutation_techniques = [partial(mutNodeReplacement, pset=self._pset)]
    mutator = np.random.choice(mutation_techniques)
    unsuccesful_mutations = 0
    for _ in range(self._max_mut_loops):
        ind = self._toolbox.clone(individual)
        offspring, = mutator(ind)
        if str(offspring) not in self.evaluated_individuals_:
            offspring.statistics['crossover_count'] = individual.statistics['crossover_count']
            offspring.statistics['mutation_count'] = individual.statistics['mutation_count'] + 1
            offspring.statistics['predecessor'] = (str(individual),)
            offspring.statistics['generation'] = 'INVALID'
            break
        else:
            unsuccesful_mutations += 1
    if unsuccesful_mutations == 50 and (type(mutator) is partial and mutator.func is gp.mutShrink):
        offspring, = self._random_mutation_operator(individual, allow_shrink=False)
    return (offspring,)