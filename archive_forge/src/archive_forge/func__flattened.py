import types
from itertools import islice
import logging
import traceback
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.deprecation import (
from .numvalue import (
from .base import ExpressionBase
from .boolean_value import BooleanValue, BooleanConstant
from .expr_common import _and, _or, _equiv, _inv, _xor, _impl, ExpressionType
from .numeric_expr import NumericExpression
import operator
def _flattened(args):
    """Flatten any potentially indexed arguments."""
    for arg in args:
        if arg.__class__ in native_types:
            yield arg
        elif isinstance(arg, (types.GeneratorType, list)):
            for _argdata in arg:
                yield _argdata
        elif arg.is_indexed():
            for _argdata in arg.values():
                yield _argdata
        else:
            yield arg