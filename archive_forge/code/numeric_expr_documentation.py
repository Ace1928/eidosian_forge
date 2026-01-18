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
A linear expression of the form `const + sum_i(c_i*x_i)`.

        You can specify `args` OR (`constant`, `linear_coefs`, and
        `linear_vars`).  If `args` is provided, it should be a list that
        contains only constants, NPV objects/expressions, or
        :py:class:`MonomialTermExpression` objects.  Alternatively, you
        can specify the constant, the list of linear_coefs and the list
        of linear_vars separately.  Note that these lists are NOT
        preserved.

        