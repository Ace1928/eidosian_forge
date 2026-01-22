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
class NumericExpression(ExpressionBase, NumericValue):
    """
    The base class for Pyomo expressions.

    This class is used to define nodes in a numeric expression
    tree.

    Args:
        args (list or tuple): Children of this node.
    """
    __slots__ = ('_args_',)
    EXPRESSION_SYSTEM = ExpressionType.NUMERIC
    PRECEDENCE = 0

    def __init__(self, args):
        self._args_ = args

    def nargs(self):
        return 2

    @property
    def args(self):
        """
        Return the child nodes

        Returns
        -------
        list or tuple:
            Sequence containing only the child nodes of this node.  The
            return type depends on the node storage model.  Users are
            not permitted to change the returned data (even for the case
            of data returned as a list), as that breaks the promise of
            tree immutability.
        """
        return self._args_

    @deprecated('The implicit recasting of a "not potentially variable" expression node to a potentially variable one is no longer supported (this violates that immutability promise for Pyomo5 expression trees).', version='6.4.3')
    def create_potentially_variable_object(self):
        """
        Create a potentially variable version of this object.

        This method returns an object that is a potentially variable
        version of the current object.  In the simplest
        case, this simply sets the value of `__class__`:

            self.__class__ = self.__class__.__mro__[1]

        Note that this method is allowed to modify the current object
        and return it.  But in some cases it may create a new
        potentially variable object.

        Returns:
            An object that is potentially variable.
        """
        if not self.is_potentially_variable():
            logger.error('recasting a non-potentially variable expression to a potentially variable one violates the immutability promise for Pyomo expression trees.')
            self.__class__ = self.potentially_variable_base_class()
        return self

    def polynomial_degree(self):
        """
        Return the polynomial degree of the expression.

        Returns:
            A non-negative integer that is the polynomial
            degree if the expression is polynomial, or :const:`None` otherwise.
        """
        return visitor.polynomial_degree(self)

    def _compute_polynomial_degree(self, values):
        """
        Compute the polynomial degree of this expression given
        the degree values of its children.

        This method is called by the :class:`_PolynomialDegreeVisitor
        <pyomo.core.expr.current._PolynomialDegreeVisitor>` class.  It can
        be over-written by expression classes to customize this
        logic.

        Args:
            values (list): A list of values that indicate the degree
                of the children expression.

        Returns:
            A nonnegative integer that is the polynomial degree of the
            expression, or :const:`None`.  Default is :const:`None`.
        """
        if all((val == 0 for val in values)):
            return 0
        else:
            return None