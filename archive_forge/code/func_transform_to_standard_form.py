import copy
from enum import Enum, auto
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.base import (
from pyomo.core.util import prod
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set_types import Reals
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import value
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.expr.visitor import (
from pyomo.common.dependencies import scipy as sp
from pyomo.core.expr.numvalue import native_types
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.core.expr.numeric_expr import SumExpression
from pyomo.environ import SolverFactory
import itertools as it
import timeit
from contextlib import contextmanager
import logging
import math
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.log import Preformatted
def transform_to_standard_form(model):
    """
    Recast all model inequality constraints of the form `a <= g(v)` (`<= b`)
    to the 'standard' form `a - g(v) <= 0` (and `g(v) - b <= 0`),
    in which `v` denotes all model variables and `a` and `b` are
    contingent on model parameters.

    Parameters
    ----------
    model : ConcreteModel
        The model to search for constraints. This will descend into all
        active Blocks and sub-Blocks as well.

    Note
    ----
    If `a` and `b` are identical and the constraint is not classified as an
    equality (i.e. the `equality` attribute of the constraint object
    is `False`), then the constraint is recast to the equality `g(v) == a`.
    """
    cons = list(model.component_data_objects(Constraint, descend_into=True, active=True))
    for con in cons:
        if not con.equality:
            has_lb = con.lower is not None
            has_ub = con.upper is not None
            if has_lb and has_ub:
                if con.lower is con.upper:
                    con.set_value(con.lower == con.body)
                else:
                    uniq_name = unique_component_name(model, con.name + '_lb')
                    model.add_component(uniq_name, Constraint(expr=con.lower - con.body <= 0))
                    con.set_value(con.body - con.upper <= 0)
            elif has_lb:
                con.set_value(con.lower - con.body <= 0)
            elif has_ub:
                con.set_value(con.body - con.upper <= 0)
            else:
                con.deactivate()