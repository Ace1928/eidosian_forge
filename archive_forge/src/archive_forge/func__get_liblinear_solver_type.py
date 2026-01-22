import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from ..base import BaseEstimator, ClassifierMixin, _fit_context
from ..exceptions import ConvergenceWarning, NotFittedError
from ..preprocessing import LabelEncoder
from ..utils import check_array, check_random_state, column_or_1d, compute_class_weight
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.metaestimators import available_if
from ..utils.multiclass import _ovr_decision_function, check_classification_targets
from ..utils.validation import (
from . import _liblinear as liblinear  # type: ignore
from . import _libsvm as libsvm  # type: ignore
from . import _libsvm_sparse as libsvm_sparse  # type: ignore
def _get_liblinear_solver_type(multi_class, penalty, loss, dual):
    """Find the liblinear magic number for the solver.

    This number depends on the values of the following attributes:
      - multi_class
      - penalty
      - loss
      - dual

    The same number is also internally used by LibLinear to determine
    which solver to use.
    """
    _solver_type_dict = {'logistic_regression': {'l1': {False: 6}, 'l2': {False: 0, True: 7}}, 'hinge': {'l2': {True: 3}}, 'squared_hinge': {'l1': {False: 5}, 'l2': {False: 2, True: 1}}, 'epsilon_insensitive': {'l2': {True: 13}}, 'squared_epsilon_insensitive': {'l2': {False: 11, True: 12}}, 'crammer_singer': 4}
    if multi_class == 'crammer_singer':
        return _solver_type_dict[multi_class]
    elif multi_class != 'ovr':
        raise ValueError('`multi_class` must be one of `ovr`, `crammer_singer`, got %r' % multi_class)
    _solver_pen = _solver_type_dict.get(loss, None)
    if _solver_pen is None:
        error_string = "loss='%s' is not supported" % loss
    else:
        _solver_dual = _solver_pen.get(penalty, None)
        if _solver_dual is None:
            error_string = "The combination of penalty='%s' and loss='%s' is not supported" % (penalty, loss)
        else:
            solver_num = _solver_dual.get(dual, None)
            if solver_num is None:
                error_string = "The combination of penalty='%s' and loss='%s' are not supported when dual=%s" % (penalty, loss, dual)
            else:
                return solver_num
    raise ValueError('Unsupported set of arguments: %s, Parameters: penalty=%r, loss=%r, dual=%r' % (error_string, penalty, loss, dual))