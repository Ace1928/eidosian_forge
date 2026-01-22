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
class _Booleans(_Constraint):
    """Constraint representing boolean likes.

    Convenience class for
    [bool, np.bool_, Integral (deprecated)]
    """

    def __init__(self):
        super().__init__()
        self._constraints = [_InstancesOf(bool), _InstancesOf(np.bool_)]

    def is_satisfied_by(self, val):
        return any((c.is_satisfied_by(val) for c in self._constraints))

    def __str__(self):
        return f'{', '.join([str(c) for c in self._constraints[:-1]])} or {self._constraints[-1]}'