import sys
import logging
import itertools
from pyomo.common.numeric_types import native_types, native_numeric_types
from pyomo.core.base import Constraint, Objective, ComponentMap
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.core.base.objective import _GeneralObjectiveData, ScalarObjective
from pyomo.core.base import _ExpressionData, Expression
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.var import ScalarVar, Var, _GeneralVarData, value
from pyomo.core.base.param import ScalarParam, _ParamData
from pyomo.core.kernel.expression import expression, noclone
from pyomo.core.kernel.variable import IVariable, variable
from pyomo.core.kernel.objective import objective
from io import StringIO
class ResultsWithoutQuadratics(object):
    __slot__ = ('const', 'nonl', 'linear')

    def __init__(self, constant=0, nonl=0, linear=None):
        self.constant = constant
        self.nonl = nonl
        self.linear = {}

    def __str__(self):
        return 'Const:\t%s\nLinear:\t%s\nNonlinear:\t%s' % (str(self.constant), str(self.linear), str(self.nonl))