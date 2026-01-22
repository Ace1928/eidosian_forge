from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.base.external import ExternalFunction
from pyomo.network import Port
from pyomo.common.collections import ComponentSet
from pyomo.repn import generate_standard_repn
import pyomo.core.expr as EXPR
from pyomo.opt import SolverFactory
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (
from pyomo.core.util import target_list
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from weakref import ref as weakref_ref
from math import floor
import logging
def arbitrary_partition(disjunction, P):
    """
    Returns a valid partition into P sets of the variables that appear in
    algebraic additively separable constraints in the Disjuncts in
    'disjunction'. Note that this method may return an invalid partition
    if the constraints are not additively separable!

    Arguments:
    ----------
    disjunction : A Disjunction object for which the variable partition will be
                 created.
    P : An int, the number of partitions
    """
    v_set = ComponentSet()
    for disj in disjunction.disjuncts:
        v_set.update(get_vars_from_components(disj, Constraint, descend_into=Block, active=True))
    partitions = [ComponentSet() for i in range(P)]
    for i, v in enumerate(v_set):
        partitions[i % P].add(v)
    return partitions