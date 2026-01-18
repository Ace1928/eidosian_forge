from numbers import Integral, Real
import numpy as np
import scipy.optimize
from ..._loss.loss import (
from ...base import BaseEstimator, RegressorMixin, _fit_context
from ...utils import check_array
from ...utils._openmp_helpers import _openmp_effective_n_threads
from ...utils._param_validation import Hidden, Interval, StrOptions
from ...utils.optimize import _check_optimize_result
from ...utils.validation import _check_sample_weight, check_is_fitted
from .._linear_loss import LinearModelLoss
from ._newton_solver import NewtonCholeskySolver, NewtonSolver
This is only necessary because of the link and power arguments of the
        TweedieRegressor.

        Note that we do not need to pass sample_weight to the loss class as this is
        only needed to set loss.constant_hessian on which GLMs do not rely.
        