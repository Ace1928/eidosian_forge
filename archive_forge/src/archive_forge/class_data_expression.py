from pyomo.common.deprecation import deprecated
from pyomo.common.modeling import NOTSET
import pyomo.core.expr as EXPR
from pyomo.core.kernel.base import ICategorizedObject, _abstract_readwrite_property
from pyomo.core.kernel.container_utils import define_simple_containers
from pyomo.core.expr.numvalue import (
class data_expression(expression):
    """A named, mutable expression that is restricted to
    storage of data expressions. An exception will be raised
    if an expression is assigned that references (or is
    allowed to reference) variables."""
    __slots__ = ()

    def is_potentially_variable(self):
        """A boolean indicating whether this expression can
        reference variables."""
        return False

    def polynomial_degree(self):
        """Always return zero because we always validate
        that the stored expression can never reference
        variables."""
        return 0

    @property
    def expr(self):
        return self._expr

    @expr.setter
    def expr(self, expr):
        if expr is not None and (not is_numeric_data(expr)):
            raise ValueError('Expression is not restricted to numeric data.')
        self._expr = expr