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
def _balanced_parens(arg):
    """Verify the string argument contains balanced parentheses.

    This checks that every open paren is balanced by a closed paren.
    That is, the infix string expression is likely to be valid.  This is
    primarily used to determine if a string that starts and ends with
    parens can have those parens removed.

    Examples:
        >>> a = "(( x + y ) * ( z - w ))"
        >>> _balanced_parens(a[1:-1])
        True
        >>> a = "( x + y ) * ( z - w )"
        >>> _balanced_parens(a[1:-1])
        False
    """
    _parenCount = 0
    for c in arg:
        if c == '(':
            _parenCount += 1
        elif c == ')':
            _parenCount -= 1
            if _parenCount < 0:
                return False
    return _parenCount == 0