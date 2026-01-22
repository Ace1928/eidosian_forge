import math
from pyomo.common.dependencies import attempt_import
from pyomo.core import value, SymbolMap, NumericLabeler, Var, Constraint
from pyomo.core.expr import (
from pyomo.core.expr.numvalue import nonpyomo_leaf_types
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.gdp import Disjunction
def _process_active_disjunction(self, djn):
    or_expr = '0'
    disjuncts = []
    for disj in djn.disjuncts:
        constraints = []
        iv = disj.binary_indicator_var
        label = self.add_var(iv)
        or_expr = '(+ ' + or_expr + ' ' + label + ')'
        for c in disj.component_data_objects(ctype=Constraint, active=True):
            try:
                constraints.append(self.walker.walk_expression(c.expr))
            except NotImplementedError as e:
                if self.logger is not None:
                    self.logger.warning('Skipping Disjunct Expression: ' + str(e))
        disjuncts.append((label, constraints))
    if djn.xor:
        or_expr = '(assert (= 1 ' + or_expr + '))\n'
    else:
        or_expr = '(assert (>= 1 ' + or_expr + '))\n'
    self.disjunctions_list.append((or_expr, disjuncts))