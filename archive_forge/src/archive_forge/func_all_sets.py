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
@all_sets.setter
def all_sets(self, val):
    if isinstance(val, dict):
        the_sets = val.values()
    else:
        the_sets = list(val)
    all_sets = UncertaintySetList(the_sets, name='all_sets', min_length=2)
    if hasattr(self, '_all_sets'):
        if all_sets.dim != self.dim:
            raise ValueError(f"Attempting to set attribute 'all_sets' of an IntersectionSet of dimension {self.dim} to a sequence of sets of dimension {all_sets[0].dim}")
    self._all_sets = all_sets