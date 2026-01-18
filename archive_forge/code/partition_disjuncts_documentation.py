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

    Transform disjunctive model to equivalent disjunctive model (with
    potentially tighter hull relaxation) by taking the "P-split" formulation
    from Kronqvist et al. 2021 [1]. In each Disjunct, convex and additively
    separable constraints are split into separate constraints by introducing
    auxiliary variables that upperbound the subexpressions created by the split.
    Increasing the number of partitions can result in tighter hull relaxations,
    but at the cost of larger model sizes.

    The transformation will create a new Block with a unique name beginning
    "_pyomo_gdp_partition_disjuncts_reformulation".
    The Block will have new Disjunct objects, each corresponding to one of the
    Disjuncts being transformed. These will have the transformed constraints on
    them, and be in new Disjunctions, each corresponding to one of the
    originals. In addition, the auxiliary variables and the partitioned
    constraints will be declared on this Block, as well as LogicalConstraints
    linking the original indicator_vars with the ones of the transformed
    Disjuncts. All original GDP components that were transformed will be
    deactivated.

    References
    ----------
        [1] J. Kronqvist, R. Misener, and C. Tsay, "Between Steps: Intermediate
            Relaxations between big-M and Convex Hull Reformulations," 2021.

    