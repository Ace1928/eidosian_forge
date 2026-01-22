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
class _ArrayLikes(_Constraint):
    """Constraint representing array-likes"""

    def is_satisfied_by(self, val):
        return _is_arraylike_not_scalar(val)

    def __str__(self):
        return 'an array-like'