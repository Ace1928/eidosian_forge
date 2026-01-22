from __future__ import annotations
from functools import wraps
import inspect
from textwrap import dedent
from typing import (
import warnings
from pandas._libs.properties import cache_readonly
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
def _deprecate_kwarg(func: F) -> F:

    @wraps(func)
    def wrapper(*args, **kwargs) -> Callable[..., Any]:
        old_arg_value = kwargs.pop(old_arg_name, None)
        if old_arg_value is not None:
            if new_arg_name is None:
                msg = f'the {repr(old_arg_name)} keyword is deprecated and will be removed in a future version. Please take steps to stop the use of {repr(old_arg_name)}'
                warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
                kwargs[old_arg_name] = old_arg_value
                return func(*args, **kwargs)
            elif mapping is not None:
                if callable(mapping):
                    new_arg_value = mapping(old_arg_value)
                else:
                    new_arg_value = mapping.get(old_arg_value, old_arg_value)
                msg = f'the {old_arg_name}={repr(old_arg_value)} keyword is deprecated, use {new_arg_name}={repr(new_arg_value)} instead.'
            else:
                new_arg_value = old_arg_value
                msg = f'the {repr(old_arg_name)} keyword is deprecated, use {repr(new_arg_name)} instead.'
            warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
            if kwargs.get(new_arg_name) is not None:
                msg = f'Can only specify {repr(old_arg_name)} or {repr(new_arg_name)}, not both.'
                raise TypeError(msg)
            kwargs[new_arg_name] = new_arg_value
        return func(*args, **kwargs)
    return cast(F, wrapper)