from pyomo.common.dependencies import attempt_import
import itertools
import logging
from operator import attrgetter
from pyomo.common import DeveloperError
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.collections import ComponentMap
from pyomo.common.fileutils import Executable
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.interval_var import (
from pyomo.contrib.cp.scheduling_expr.precedence_expressions import (
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.core.base import (
from pyomo.core.base.boolean_var import (
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.param import IndexedParam, ScalarParam, _ParamData
from pyomo.core.base.var import ScalarVar, _GeneralVarData, IndexedVar
import pyomo.core.expr as EXPR
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, identify_variables
from pyomo.core.base import Set, RangeSet
from pyomo.core.base.set import SetProduct
from pyomo.opt import WriterFactory, SolverFactory, TerminationCondition, SolverResults
from pyomo.network import Port
def _handle_getattr(visitor, node, obj, attr):
    if obj[0] is _DEFERRED_ELEMENT_CONSTRAINT:
        try:
            ans = list(map(_deferred_element_getattr_dispatcher[attr[1]], obj[1][0]))
        except KeyError:
            logger.error('Unrecognized attribute in GetAttrExpression: %s.' % attr[1])
            raise
        return (_ELEMENT_CONSTRAINT, cp.element(array=ans, index=obj[1][1]))
    elif obj[0] is _ELEMENT_CONSTRAINT:
        try:
            return (_element_constraint_attr_dispatcher[attr[1]], obj)
        except KeyError:
            logger.error('Unrecognized attribute in GetAttrExpression:%s. Found for object: %s' % (attr[1], obj[1]))
            raise
    else:
        raise DeveloperError("Unrecognized argument type '%s' to getattr dispatcher." % obj[0])