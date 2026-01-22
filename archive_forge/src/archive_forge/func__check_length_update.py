import abc
import math
import functools
from numbers import Integral
from collections.abc import Iterable, MutableSequence
from enum import Enum
from pyomo.common.dependencies import numpy as np, scipy as sp
from pyomo.core.base import ConcreteModel, Objective, maximize, minimize, Block
from pyomo.core.base.constraint import ConstraintList
from pyomo.core.base.var import Var, IndexedVar
from pyomo.core.expr.numvalue import value, native_numeric_types
from pyomo.opt.results import check_optimal_termination
from pyomo.contrib.pyros.util import add_bounds_for_uncertain_parameters
def _check_length_update(self, idx, value):
    """
        Check whether the update ``self[idx] = value`` reduces the
        length of self to a value smaller than the minimum length.

        Raises
        ------
        ValueError
            If minimum length requirement is violated by the update.
        """
    if isinstance(idx, Integral):
        slice_len = 1
    else:
        slice_len = len(self._list[idx])
    val_len = len(value) if isinstance(value, Iterable) else 1
    new_len = len(self) + val_len - slice_len
    if new_len < self._min_length:
        raise ValueError(f'Length of uncertainty set list {self._name!r} must be at least {self._min_length}')