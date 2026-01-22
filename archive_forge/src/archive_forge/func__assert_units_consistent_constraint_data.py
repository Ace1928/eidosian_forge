import logging
from pyomo.core.base.units_container import units, UnitsError
from pyomo.core.base import (
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.network import Port, Arc
from pyomo.mpec import Complementarity
from pyomo.gdp import Disjunct, Disjunction
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.numvalue import native_types
from pyomo.util.components import iter_component
from pyomo.common.collections import ComponentSet
def _assert_units_consistent_constraint_data(condata):
    """
    Raise an exception if the any units in lower, body, upper on a
    ConstraintData object are not consistent or are not equivalent
    with each other.
    """
    args = list()
    if condata.lower is not None and value(condata.lower) != 0.0:
        args.append(condata.lower)
    args.append(condata.body)
    if condata.upper is not None and value(condata.upper) != 0.0:
        args.append(condata.upper)
    if len(args) == 1:
        assert_units_consistent(*args)
    else:
        assert_units_equivalent(*args)