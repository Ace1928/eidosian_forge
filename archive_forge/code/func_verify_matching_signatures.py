import collections
import functools
import os
from .._utils import set_module
from .._utils._inspect import getargspec
from numpy.core._multiarray_umath import (
def verify_matching_signatures(implementation, dispatcher):
    """Verify that a dispatcher function has the right signature."""
    implementation_spec = ArgSpec(*getargspec(implementation))
    dispatcher_spec = ArgSpec(*getargspec(dispatcher))
    if implementation_spec.args != dispatcher_spec.args or implementation_spec.varargs != dispatcher_spec.varargs or implementation_spec.keywords != dispatcher_spec.keywords or (bool(implementation_spec.defaults) != bool(dispatcher_spec.defaults)) or (implementation_spec.defaults is not None and len(implementation_spec.defaults) != len(dispatcher_spec.defaults)):
        raise RuntimeError('implementation and dispatcher for %s have different function signatures' % implementation)
    if implementation_spec.defaults is not None:
        if dispatcher_spec.defaults != (None,) * len(dispatcher_spec.defaults):
            raise RuntimeError('dispatcher functions can only use None for default argument values')