from __future__ import annotations
import functools
import inspect
from typing import (
import numpy as np
from pandas.compat._optional import import_optional_dependency
from pandas.core.util.numba_ import (
def validate_udf(func: Callable) -> None:
    """
    Validate user defined function for ops when using Numba with groupby ops.

    The first signature arguments should include:

    def f(values, index, ...):
        ...

    Parameters
    ----------
    func : function, default False
        user defined function

    Returns
    -------
    None

    Raises
    ------
    NumbaUtilError
    """
    if not callable(func):
        raise NotImplementedError('Numba engine can only be used with a single function.')
    udf_signature = list(inspect.signature(func).parameters.keys())
    expected_args = ['values', 'index']
    min_number_args = len(expected_args)
    if len(udf_signature) < min_number_args or udf_signature[:min_number_args] != expected_args:
        raise NumbaUtilError(f'The first {min_number_args} arguments to {func.__name__} must be {expected_args}')