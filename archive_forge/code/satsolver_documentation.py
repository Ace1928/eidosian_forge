import math
from pyomo.common.dependencies import attempt_import
from pyomo.core import value, SymbolMap, NumericLabeler, Var, Constraint
from pyomo.core.expr import (
from pyomo.core.expr.numvalue import nonpyomo_leaf_types
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.gdp import Disjunction

        Defined string representation of object
        