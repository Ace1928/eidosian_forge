import math
from pyomo.common.dependencies import attempt_import
from pyomo.core import value, SymbolMap, NumericLabeler, Var, Constraint
from pyomo.core.expr import (
from pyomo.core.expr.numvalue import nonpyomo_leaf_types
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.gdp import Disjunction
def _process_model(self, model):
    for v in model.component_data_objects(ctype=Var, descend_into=True):
        smtstring = self.add_var(v)
    for c in model.component_data_objects(ctype=Constraint, active=True):
        self.add_expr(c.expr)
    for djn in model.component_data_objects(ctype=Disjunction):
        if djn.active:
            self._process_active_disjunction(djn)
        else:
            self._process_inactive_disjunction(djn)