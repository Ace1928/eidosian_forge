import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def _pint_convert_temp_from_to(self, numerical_value, pint_from_units, pint_to_units):
    if type(numerical_value) not in native_numeric_types:
        raise UnitsError('Conversion routines for absolute and relative temperatures require a numerical value only. Pyomo objects (Var, Param, expressions) are not supported. Please use value(x) to extract the numerical value if necessary.')
    src_quantity = self._pint_registry.Quantity(numerical_value, pint_from_units)
    dest_quantity = src_quantity.to(pint_to_units)
    return dest_quantity.magnitude