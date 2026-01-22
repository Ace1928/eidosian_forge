import inspect
import pprint
from collections import OrderedDict
from .._config import get_config
from ..base import BaseEstimator
from . import is_scalar_nan
def _changed_params(estimator):
    """Return dict (param_name: value) of parameters that were given to
    estimator with non-default values."""
    params = estimator.get_params(deep=False)
    init_func = getattr(estimator.__init__, 'deprecated_original', estimator.__init__)
    init_params = inspect.signature(init_func).parameters
    init_params = {name: param.default for name, param in init_params.items()}

    def has_changed(k, v):
        if k not in init_params:
            return True
        if init_params[k] == inspect._empty:
            return True
        if isinstance(v, BaseEstimator) and v.__class__ != init_params[k].__class__:
            return True
        if repr(v) != repr(init_params[k]) and (not (is_scalar_nan(init_params[k]) and is_scalar_nan(v))):
            return True
        return False
    return {k: v for k, v in params.items() if has_changed(k, v)}