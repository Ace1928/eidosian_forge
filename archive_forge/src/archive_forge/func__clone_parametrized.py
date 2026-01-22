import copy
import functools
import inspect
import platform
import re
import warnings
from collections import defaultdict
import numpy as np
from . import __version__
from ._config import config_context, get_config
from .exceptions import InconsistentVersionWarning
from .utils import _IS_32BIT
from .utils._estimator_html_repr import _HTMLDocumentationLinkMixin, estimator_html_repr
from .utils._metadata_requests import _MetadataRequester, _routing_enabled
from .utils._param_validation import validate_parameter_constraints
from .utils._set_output import _SetOutputMixin
from .utils._tags import (
from .utils.validation import (
def _clone_parametrized(estimator, *, safe=True):
    """Default implementation of clone. See :func:`sklearn.base.clone` for details."""
    estimator_type = type(estimator)
    if estimator_type is dict:
        return {k: clone(v, safe=safe) for k, v in estimator.items()}
    elif estimator_type in (list, tuple, set, frozenset):
        return estimator_type([clone(e, safe=safe) for e in estimator])
    elif not hasattr(estimator, 'get_params') or isinstance(estimator, type):
        if not safe:
            return copy.deepcopy(estimator)
        elif isinstance(estimator, type):
            raise TypeError('Cannot clone object. ' + 'You should provide an instance of ' + 'scikit-learn estimator instead of a class.')
        else:
            raise TypeError("Cannot clone object '%s' (type %s): it does not seem to be a scikit-learn estimator as it does not implement a 'get_params' method." % (repr(estimator), type(estimator)))
    klass = estimator.__class__
    new_object_params = estimator.get_params(deep=False)
    for name, param in new_object_params.items():
        new_object_params[name] = clone(param, safe=False)
    new_object = klass(**new_object_params)
    try:
        new_object._metadata_request = copy.deepcopy(estimator._metadata_request)
    except AttributeError:
        pass
    params_set = new_object.get_params(deep=False)
    for name in new_object_params:
        param1 = new_object_params[name]
        param2 = params_set[name]
        if param1 is not param2:
            raise RuntimeError('Cannot clone object %s, as the constructor either does not set or modifies parameter %s' % (estimator, name))
    if hasattr(estimator, '_sklearn_output_config'):
        new_object._sklearn_output_config = copy.deepcopy(estimator._sklearn_output_config)
    return new_object