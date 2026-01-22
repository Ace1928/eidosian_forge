import math
import copy
import re
import io
import pyomo.environ as pyo
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.expr import (
from pyomo.core.expr.visitor import identify_components
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.objective import ScalarObjective, _GeneralObjectiveData
import pyomo.core.kernel as kernel
from pyomo.core.expr.template_expr import (
from pyomo.core.base.var import ScalarVar, _GeneralVarData, IndexedVar
from pyomo.core.base.param import _ParamData, ScalarParam, IndexedParam
from pyomo.core.base.set import _SetData
from pyomo.core.base.constraint import ScalarConstraint, IndexedConstraint
from pyomo.common.collections.component_map import ComponentMap
from pyomo.common.collections.component_set import ComponentSet
from pyomo.core.expr.template_expr import (
from pyomo.core.expr.numeric_expr import NPV_SumExpression, NPV_DivisionExpression
from pyomo.core.base.block import IndexedBlock
from pyomo.core.base.external import _PythonCallbackFunctionID
from pyomo.core.base.enums import SortComponents
from pyomo.core.base.block import _BlockData
from pyomo.repn.util import ExprType
from pyomo.common import DeveloperError
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.common.dependencies import numpy as np, numpy_available
def analyze_variable(vr):
    domainMap = {'Reals': '\\mathds{R}', 'PositiveReals': '\\mathds{R}_{> 0}', 'NonPositiveReals': '\\mathds{R}_{\\leq 0}', 'NegativeReals': '\\mathds{R}_{< 0}', 'NonNegativeReals': '\\mathds{R}_{\\geq 0}', 'Integers': '\\mathds{Z}', 'PositiveIntegers': '\\mathds{Z}_{> 0}', 'NonPositiveIntegers': '\\mathds{Z}_{\\leq 0}', 'NegativeIntegers': '\\mathds{Z}_{< 0}', 'NonNegativeIntegers': '\\mathds{Z}_{\\geq 0}', 'Boolean': '\\left\\{ \\text{True} , \\text{False} \\right \\}', 'Binary': '\\left\\{ 0 , 1 \\right \\}', 'EmptySet': '\\varnothing', 'UnitInterval': '\\mathds{R}', 'PercentFraction': '\\mathds{R}'}
    domainName = vr.domain.name
    varBounds = vr.bounds
    lowerBoundValue = varBounds[0]
    upperBoundValue = varBounds[1]
    if domainName in ['Reals', 'Integers']:
        if lowerBoundValue is not None:
            lowerBound = str(lowerBoundValue) + ' \\leq '
        else:
            lowerBound = ''
        if upperBoundValue is not None:
            upperBound = ' \\leq ' + str(upperBoundValue)
        else:
            upperBound = ''
    elif domainName in ['PositiveReals', 'PositiveIntegers']:
        if lowerBoundValue > 0:
            lowerBound = str(lowerBoundValue) + ' \\leq '
        else:
            lowerBound = ' 0 < '
        if upperBoundValue is not None:
            if upperBoundValue <= 0:
                raise InfeasibleConstraintException('Formulation is infeasible due to bounds on variable %s' % vr.name)
            else:
                upperBound = ' \\leq ' + str(upperBoundValue)
        else:
            upperBound = ''
    elif domainName in ['NonPositiveReals', 'NonPositiveIntegers']:
        if lowerBoundValue is not None:
            if lowerBoundValue > 0:
                raise InfeasibleConstraintException('Formulation is infeasible due to bounds on variable %s' % vr.name)
            elif lowerBoundValue == 0:
                lowerBound = ' 0 = '
            else:
                lowerBound = str(lowerBoundValue) + ' \\leq '
        else:
            lowerBound = ''
        if upperBoundValue >= 0:
            upperBound = ' \\leq 0 '
        else:
            upperBound = ' \\leq ' + str(upperBoundValue)
    elif domainName in ['NegativeReals', 'NegativeIntegers']:
        if lowerBoundValue is not None:
            if lowerBoundValue >= 0:
                raise InfeasibleConstraintException('Formulation is infeasible due to bounds on variable %s' % vr.name)
            else:
                lowerBound = str(lowerBoundValue) + ' \\leq '
        else:
            lowerBound = ''
        if upperBoundValue >= 0:
            upperBound = ' < 0 '
        else:
            upperBound = ' \\leq ' + str(upperBoundValue)
    elif domainName in ['NonNegativeReals', 'NonNegativeIntegers']:
        if lowerBoundValue > 0:
            lowerBound = str(lowerBoundValue) + ' \\leq '
        else:
            lowerBound = ' 0 \\leq '
        if upperBoundValue is not None:
            if upperBoundValue < 0:
                raise InfeasibleConstraintException('Formulation is infeasible due to bounds on variable %s' % vr.name)
            elif upperBoundValue == 0:
                upperBound = ' = 0 '
            else:
                upperBound = ' \\leq ' + str(upperBoundValue)
        else:
            upperBound = ''
    elif domainName in ['Boolean', 'Binary', 'Any', 'AnyWithNone', 'EmptySet']:
        lowerBound = ''
        upperBound = ''
    elif domainName in ['UnitInterval', 'PercentFraction']:
        if lowerBoundValue > 1:
            raise InfeasibleConstraintException('Formulation is infeasible due to bounds on variable %s' % vr.name)
        elif lowerBoundValue == 1:
            lowerBound = ' = 1 '
        elif lowerBoundValue > 0:
            lowerBound = str(lowerBoundValue) + ' \\leq '
        else:
            lowerBound = ' 0 \\leq '
        if upperBoundValue < 0:
            raise InfeasibleConstraintException('Formulation is infeasible due to bounds on variable %s' % vr.name)
        elif upperBoundValue == 0:
            upperBound = ' = 0 '
        elif upperBoundValue < 1:
            upperBound = ' \\leq ' + str(upperBoundValue)
        else:
            upperBound = ' \\leq 1 '
    else:
        raise NotImplementedError('Invalid domain encountered, will be supported in a future update')
    varBoundData = {'variable': vr, 'lowerBound': lowerBound, 'upperBound': upperBound, 'domainName': domainName, 'domainLatex': domainMap[domainName]}
    return varBoundData