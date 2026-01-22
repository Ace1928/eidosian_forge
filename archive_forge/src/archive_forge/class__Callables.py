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
class _Callables(_Constraint):
    """Constraint representing callables."""

    def is_satisfied_by(self, val):
        return callable(val)

    def __str__(self):
        return 'a callable'