import collections
import enum
import logging
import math
import operator
from pyomo.common.dependencies import attempt_import
from pyomo.common.deprecation import deprecated, relocated_module_attribute
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.formatting import tostr
from pyomo.common.numeric_types import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.expr_common import (
from pyomo.core.expr.base import ExpressionBase, NPV_Mixin, visitor
class mutable_expression(object):
    """Context manager for mutable sums.

    This context manager is used to compute a sum while treating the
    summation as a mutable object.

    """

    def __enter__(self):
        self.e = _MutableNPVSumExpression([])
        return self.e

    def __exit__(self, *args):
        if isinstance(self.e, _MutableSumExpression):
            self.e.make_immutable()