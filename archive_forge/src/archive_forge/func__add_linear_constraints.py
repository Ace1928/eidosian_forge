from pyomo.core import (
from pyomo.core.base import TransformationFactory, _VarData
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.common.config import ConfigBlock, ConfigValue, NonNegativeFloat
from pyomo.common.modeling import unique_component_name
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.opt import TerminationCondition
import logging
def _add_linear_constraints(self, cons1, cons2):
    """Adds two >= constraints

        Because this is always called after
        _nonneg_scalar_multiply_linear_constraint, though it is implemented
        more generally.
        """
    ans = {'lower': None, 'body': None, 'map': ComponentMap()}
    cons1_body = cons1['body']
    cons2_body = cons2['body']
    all_vars = list(cons1_body.linear_vars)
    seen = ComponentSet(all_vars)
    for v in cons2_body.linear_vars:
        if v not in seen:
            all_vars.append(v)
    expr = 0
    for var in all_vars:
        coef = self._add(cons1['map'].get(var, 0), cons2['map'].get(var, 0), self._add_linear_constraints_error_msg, (cons1, cons2))
        ans['map'][var] = coef
        expr += coef * var
    for cons in [cons1_body, cons2_body]:
        if cons.nonlinear_expr is not None:
            expr += cons.nonlinear_expr
        expr += sum((coef * v1 * v2 for coef, (v1, v2) in zip(cons.quadratic_coefs, cons.quadratic_vars)))
    ans['body'] = generate_standard_repn(expr)
    ans['lower'] = self._add(cons1['lower'], cons2['lower'], self._add_linear_constraints_error_msg, (cons1, cons2))
    return ans