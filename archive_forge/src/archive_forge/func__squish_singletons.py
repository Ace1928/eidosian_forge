import copy
import itertools
from pyomo.core import Block, ConstraintList, Set, Constraint
from pyomo.core.base import Reference
from pyomo.common.modeling import unique_component_name
from pyomo.gdp.disjunct import Disjunct, Disjunction
import logging
def _squish_singletons(tuple_iter):
    """Squish all singleton tuples into their non-tuple values."""
    for tup in tuple_iter:
        if len(tup) == 1:
            yield tup[0]
        else:
            yield tup