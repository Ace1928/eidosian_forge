from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.transformation import TransformationFactory
from pyomo.core.base.suffix import Suffix
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.common.config import (
from pyomo.common.errors import InfeasibleConstraintException
def _build_equality_set(m):
    """Construct an equality set map.

    Maps all variables to the set of variables that are linked to them by
    equality. Mapping takes place using id(). That is, if you have x = y, then
    you would have id(x) -> ComponentSet([x, y]) and id(y) -> ComponentSet([x,
    y]) in the mapping.

    """
    eq_var_map = ComponentMap()
    relevant_vars = ComponentSet()
    for constr in m.component_data_objects(ctype=Constraint, active=True, descend_into=True):
        if value(constr.lower) == 0 and value(constr.upper) == 0 and (constr.body.polynomial_degree() == 1):
            repn = generate_standard_repn(constr.body)
            vars_ = [v for i, v in enumerate(repn.linear_vars) if repn.linear_coefs[i]]
            if len(vars_) == 2 and repn.constant == 0 and (sorted((l for l in repn.linear_coefs if l)) == [-1, 1]):
                v1 = vars_[0]
                v2 = vars_[1]
                set1 = eq_var_map.get(v1, ComponentSet([v1]))
                set2 = eq_var_map.get(v2, ComponentSet([v2]))
                relevant_vars.update([v1, v2])
                set1.update(set2)
                for v in set1:
                    eq_var_map[v] = set1
    return (eq_var_map, relevant_vars)