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
class ExternalFunctionExpression(NumericExpression):
    """
    External function expressions

    Example::

        model = ConcreteModel()
        model.a = Var()
        model.f = ExternalFunction(library='foo.so', function='bar')
        expr = model.f(model.a)

    Args:
        args (tuple): children of this node
        fcn: a class that defines this external function
    """
    __slots__ = ('_fcn',)
    PRECEDENCE = None

    def __init__(self, args, fcn=None):
        self._args_ = args
        self._fcn = fcn

    def nargs(self):
        return len(self._args_)

    def create_node_with_local_data(self, args, classtype=None):
        if classtype is None:
            classtype = self.__class__
        return classtype(args, self._fcn)

    def getname(self, *args, **kwds):
        return self._fcn.getname(*args, **kwds)

    def _apply_operation(self, result):
        return self._fcn.evaluate(result)

    def _to_string(self, values, verbose, smap):
        return f'{self.getname()}({', '.join(values)})'

    def get_arg_units(self):
        """Return the units for this external functions arguments"""
        return self._fcn.get_arg_units()

    def get_units(self):
        """Get the units of the return value for this external function"""
        return self._fcn.get_units()