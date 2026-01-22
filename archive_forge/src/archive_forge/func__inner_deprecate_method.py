import warnings
from functools import wraps
from inspect import Parameter, signature
from typing import Iterable, Optional
def _inner_deprecate_method(f):
    name = f.__name__
    if name == '__init__':
        name = f.__qualname__.split('.')[0]

    @wraps(f)
    def inner_f(*args, **kwargs):
        warning_message = f"'{name}' (from '{f.__module__}') is deprecated and will be removed from version '{version}'."
        if message is not None:
            warning_message += ' ' + message
        warnings.warn(warning_message, FutureWarning)
        return f(*args, **kwargs)
    return inner_f