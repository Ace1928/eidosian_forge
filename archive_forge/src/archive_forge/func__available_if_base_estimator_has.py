from abc import ABCMeta, abstractmethod
from numbers import Integral
import numpy as np
import scipy.sparse as sp
from .base import (
from .model_selection import cross_val_predict
from .utils import Bunch, _print_elapsed_time, check_random_state
from .utils._param_validation import HasMethods, StrOptions
from .utils.metadata_routing import (
from .utils.metaestimators import available_if
from .utils.multiclass import check_classification_targets
from .utils.parallel import Parallel, delayed
from .utils.validation import _check_method_params, check_is_fitted, has_fit_parameter
def _available_if_base_estimator_has(attr):
    """Return a function to check if `base_estimator` or `estimators_` has `attr`.

    Helper for Chain implementations.
    """

    def _check(self):
        return hasattr(self.base_estimator, attr) or all((hasattr(est, attr) for est in self.estimators_))
    return available_if(_check)