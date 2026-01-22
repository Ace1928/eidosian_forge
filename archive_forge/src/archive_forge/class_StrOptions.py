import functools
import math
import operator
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from inspect import signature
from numbers import Integral, Real
import numpy as np
from scipy.sparse import csr_matrix, issparse
from .._config import config_context, get_config
from .validation import _is_arraylike_not_scalar
class StrOptions(Options):
    """Constraint representing a finite set of strings.

    Parameters
    ----------
    options : set of str
        The set of valid strings.

    deprecated : set of str or None, default=None
        A subset of the `options` to mark as deprecated in the string
        representation of the constraint.
    """

    def __init__(self, options, *, deprecated=None):
        super().__init__(type=str, options=options, deprecated=deprecated)