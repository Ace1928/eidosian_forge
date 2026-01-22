from pyomo.core.expr.numvalue import is_numeric_data
from pyomo.core.expr import value, exp
from pyomo.core.kernel.block import block
from pyomo.core.kernel.variable import IVariable, variable, variable_tuple
from pyomo.core.kernel.constraint import (
class _ConicBase(IConstraint):
    """Base class for a few conic constraints that
    implements some shared functionality. Derived classes
    are expected to declare any necessary slots."""
    _ctype = IConstraint
    _linear_canonical_form = False
    __slots__ = ()

    def __init__(self):
        self._parent = None
        self._storage_key = None
        self._active = True
        self._body = None

    @classmethod
    def as_domain(cls, *args, **kwds):
        """Builds a conic domain"""
        raise NotImplementedError

    def _body_function(self, *args):
        """A function that defines the body expression"""
        raise NotImplementedError

    def _body_function_variables(self, values=False):
        """Returns variables in the order they should be
        passed to the body function. If values is True, then
        return the current value of each variable in place
        of the variables themselves."""
        raise NotImplementedError

    def check_convexity_conditions(self, relax=False):
        """Returns True if all convexity conditions for the
        conic constraint are satisfied. If relax is True,
        then variable domains are ignored and it is assumed
        that all variables are continuous."""
        raise NotImplementedError

    @property
    def body(self):
        """The body of the constraint"""
        if self._body is None:
            self._body = self._body_function(*self._body_function_variables(values=False))
        return self._body

    @property
    def lower(self):
        """The expression for the lower bound of the constraint"""
        return None

    @property
    def upper(self):
        """The expression for the upper bound of the constraint"""
        return 0.0

    @property
    def lb(self):
        """The value of the lower bound of the constraint"""
        return None

    @property
    def ub(self):
        """The value of the upper bound of the constraint"""
        return 0.0

    @property
    def rhs(self):
        """The right-hand side of the constraint"""
        raise ValueError('The rhs property can not be read because this is not an equality constraint')

    @property
    def equality(self):
        return False

    def __call__(self, exception=True):
        try:
            return value(self._body_function(*self._body_function_variables(values=True)))
        except (ValueError, TypeError):
            if exception:
                raise ValueError('one or more terms could not be evaluated')
            return None