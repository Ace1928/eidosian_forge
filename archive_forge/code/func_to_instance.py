import datetime
import importlib
import inspect
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import six  # type: ignore
from triad.utils.assertion import assert_or_throw
def to_instance(s: Any, expected_base_type: Optional[type]=None, args: List[Any]=EMPTY_ARGS, kwargs: Dict[str, Any]=EMPTY_KWARGS, global_vars: Optional[Dict[str, Any]]=None, local_vars: Optional[Dict[str, Any]]=None) -> Any:
    """If s is str or type, then use :func:`~triad.utils.convert.to_type` to find
    matching type and instantiate. Otherwise return s if it matches constraints

    :param s: see :func:`~triad.utils.convert.to_type`
    :param expected_base_type: see :func:`~triad.utils.convert.to_type`
    :param args: args to instantiate the type
    :param kwargs: kwargs to instantiate the type
    :param global_vars: overriding global variables, if None, it will
      use the caller's globals(), defaults to None
    :param local_vars: overriding local variables, if None, it will
      use the caller's locals(), defaults to None

    :raises ValueError: if s is an instance but not a (sub)type of `expected_base_type`
    :raises TypeError: if s is an instance, args and kwargs must be empty

    :return: the instantiated object
    """
    global_vars, local_vars = get_caller_global_local_vars(global_vars, local_vars)
    if s is None:
        raise ValueError("None can't be converted to instance")
    if isinstance(s, (str, type)):
        t = to_type(s, expected_base_type, global_vars, local_vars)
        return t(*args, **kwargs)
    else:
        if expected_base_type is not None and (not isinstance(s, expected_base_type)):
            raise TypeError(f'{str(s)} is not a subclass of {str(expected_base_type)}')
        if len(args) > 0 or len(kwargs) > 0:
            raise ValueError(f"Can't instantiate {str(s)} with different parameters")
        return s