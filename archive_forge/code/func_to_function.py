import datetime
import importlib
import inspect
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import six  # type: ignore
from triad.utils.assertion import assert_or_throw
def to_function(func: Any, global_vars: Optional[Dict[str, Any]]=None, local_vars: Optional[Dict[str, Any]]=None) -> Any:
    """For an expression, it tries to find the matching function.

    :params s: a string expression or a callable
    :param global_vars: overriding global variables, if None, it will
      use the caller's globals(), defaults to None
    :param local_vars: overriding local variables, if None, it will
      use the caller's locals(), defaults to None

    :raises AttributeError: if unable to find such a function

    :return: the matching function
    """
    if isinstance(func, str):
        global_vars, local_vars = get_caller_global_local_vars(global_vars, local_vars)
        try:
            func = str_to_object(func, global_vars, local_vars)
        except ValueError:
            raise AttributeError(f'{func} is not a function')
    assert_or_throw(callable(func) and (not isinstance(func, six.class_types)), AttributeError(f'{func} is not a function'))
    return func