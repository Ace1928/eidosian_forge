import re
from contextlib import contextmanager
import functools
import operator
import warnings
import numbers
from collections import namedtuple
import inspect
import math
from typing import (
import numpy as np
from scipy._lib._array_api import array_namespace
def _validate_int(k, name, minimum=None):
    """
    Validate a scalar integer.

    This function can be used to validate an argument to a function
    that expects the value to be an integer.  It uses `operator.index`
    to validate the value (so, for example, k=2.0 results in a
    TypeError).

    Parameters
    ----------
    k : int
        The value to be validated.
    name : str
        The name of the parameter.
    minimum : int, optional
        An optional lower bound.
    """
    try:
        k = operator.index(k)
    except TypeError:
        raise TypeError(f'{name} must be an integer.') from None
    if minimum is not None and k < minimum:
        raise ValueError(f'{name} must be an integer not less than {minimum}') from None
    return k