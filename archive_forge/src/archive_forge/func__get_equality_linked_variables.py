from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.core.base import Block, Constraint, VarList, Objective, TransformationFactory
from pyomo.core.expr import ExpressionReplacementVisitor
from pyomo.core.expr.numvalue import value
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.repn import generate_standard_repn
import logging
def _get_equality_linked_variables(constraint):
    """Return the two variables linked by an equality constraint x == y.

    If the constraint does not match this form, skip it.

    """
    if value(constraint.lower) != 0 or value(constraint.upper) != 0:
        return ()
    if constraint.body.polynomial_degree() != 1:
        return ()
    repn = generate_standard_repn(constraint.body)
    nonzero_coef_vars = tuple((v for i, v in enumerate(repn.linear_vars) if repn.linear_coefs[i] != 0))
    if len(nonzero_coef_vars) != 2:
        return ()
    if sorted((coef for coef in repn.linear_coefs if coef != 0)) != [-1, 1]:
        return ()
    return nonzero_coef_vars