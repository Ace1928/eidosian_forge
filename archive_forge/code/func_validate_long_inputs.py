import collections
import hashlib
from functools import wraps
import flask
from .dependencies import (
from .exceptions import (
from ._grouping import (
from ._utils import (
from . import _validate
from .long_callback.managers import BaseLongCallbackManager
from ._callback_context import context_value
def validate_long_inputs(deps):
    for dep in deps:
        if dep.has_wildcard():
            raise WildcardInLongCallback(f'\n                long callbacks does not support dependencies with\n                pattern-matching ids\n                    Received: {repr(dep)}\n')