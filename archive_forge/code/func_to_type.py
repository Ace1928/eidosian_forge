import datetime
import importlib
import inspect
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import six  # type: ignore
from triad.utils.assertion import assert_or_throw
def to_type(s: Any, expected_base_type: Optional[type]=None, global_vars: Optional[Dict[str, Any]]=None, local_vars: Optional[Dict[str, Any]]=None) -> type:
    """Convert an object `s` to `type`
    * if `s` is `str`: see :func:`~triad.utils.convert.str_to_type`
    * if `s` is `type`: check `expected_base_type` and return itself
    * else: check `expected_base_type` and return itself

    :param s: see :func:`~triad.utils.convert.str_to_type`
    :param expected_base_type: see :func:`~triad.utils.convert.str_to_type`
    :param global_vars: overriding global variables, if None, it will
      use the caller's globals(), defaults to None
    :param local_vars: overriding local variables, if None, it will
      use the caller's locals(), defaults to None

    :raises TypeError: if no matching type found

    :return: the matching type
    """
    global_vars, local_vars = get_caller_global_local_vars(global_vars, local_vars)
    if isinstance(s, str):
        return str_to_type(s, expected_base_type, global_vars, local_vars)
    if isinstance(s, type):
        if expected_base_type is None or issubclass(s, expected_base_type):
            return s
        raise TypeError(f'Type mismatch {s} expected {expected_base_type}')
    t = type(s)
    if expected_base_type is None or issubclass(t, expected_base_type):
        return t
    raise TypeError(f'Type mismatch {s} expected {expected_base_type}')