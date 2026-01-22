import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def _pint_registry_mapper(encode, val):
    if encode:
        if val is not units._pint_registry:
            logger.warning('pickling a _PyomoUnit associated with a PyomoUnitsContainer that is not the default singleton (%s.units).  Restoring this state will attempt to return a unit associated with the default singleton.' % (__name__,))
        return None
    elif val is None:
        return units._pint_registry
    else:
        return val