from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.expr import identify_variables
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import is_child_of, get_gdp_tree
from pyomo.repn.standard_repn import generate_standard_repn
import logging

    Implements a special case of the transformation mentioned in [1] for
    handling disjunctive constraints with common left-hand sides (i.e.,
    Constraint bodies). Automatically detects univariate disjunctive
    Constraints (bounds or equalities involving one variable), and
    transforms them according to [1]. The transformed Constraints are
    deactivated, but the remainder of the GDP is untouched. That is,
    to completely transform the GDP, a GDP-to-MIP transformation is
    needed that will transform the remaining disjunctive constraints as
    well as any LogicalConstraints and the logic of the disjunctions
    themselves.

    NOTE: Because this transformation allows tighter bound values higher in
    the GDP hierarchy to supersede looser ones that are lower, the transformed
    model will not necessarily still be valid in the case that there are
    mutable Params in disjunctive variable bounds or in the transformed
    Constraints and the values of those mutable Params are later changed.
    Similarly, if this transformation is called when Vars are fixed, it will
    only be guaranteed to be valid when those Vars remain fixed to the same
    values.

    [1] Egon Balas, "On the convex hull of the union of certain polyhedra,"
        Operations Research Letters, vol. 7, 1988, pp. 279-283
    