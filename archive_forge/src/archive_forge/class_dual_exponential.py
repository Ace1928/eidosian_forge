from pyomo.core.expr.numvalue import is_numeric_data
from pyomo.core.expr import value, exp
from pyomo.core.kernel.block import block
from pyomo.core.kernel.variable import IVariable, variable, variable_tuple
from pyomo.core.kernel.constraint import (
class dual_exponential(_ConicBase):
    """A dual exponential conic constraint of the form:

        -x2*exp((x1/x2)-1) <= r

    which is recognized as convex for x2 <= 0 and r >= 0.

    Parameters
    ----------
    r : :class:`variable`
        A variable.
    x1 : :class:`variable`
        A variable.
    x2 : :class:`variable`
        A variable.
    """
    __slots__ = ('_parent', '_storage_key', '_active', '_body', '_r', '_x1', '_x2', '__weakref__')

    def __init__(self, r, x1, x2):
        super(dual_exponential, self).__init__()
        self._r = r
        self._x1 = x1
        self._x2 = x2
        assert isinstance(self._r, IVariable)
        assert isinstance(self._x1, IVariable)
        assert isinstance(self._x2, IVariable)

    @classmethod
    def as_domain(cls, r, x1, x2):
        """Builds a conic domain. Input arguments take the
        same form as those of the conic constraint, but in
        place of each variable, one can optionally supply a
        constant, linear expression, or None.

        Returns
        -------
        block
            A block object with the core conic constraint
            (block.q) expressed using auxiliary variables
            (block.r, block.x1, block.x2) linked to the
            input arguments through auxiliary constraints
            (block.c).
        """
        b = block()
        b.r = variable(lb=0)
        b.x1 = variable()
        b.x2 = variable(ub=0)
        b.c = _build_linking_constraints([r, x1, x2], [b.r, b.x1, b.x2])
        b.q = cls(r=b.r, x1=b.x1, x2=b.x2)
        return b

    @property
    def r(self):
        return self._r

    @property
    def x1(self):
        return self._x1

    @property
    def x2(self):
        return self._x2

    def _body_function(self, r, x1, x2):
        """A function that defines the body expression"""
        return -x2 * exp(x1 / x2 - 1) - r

    def _body_function_variables(self, values=False):
        """Returns variables in the order they should be
        passed to the body function. If values is True, then
        return the current value of each variable in place
        of the variables themselves."""
        if not values:
            return (self.r, self.x1, self.x2)
        else:
            return (self.r.value, self.x1.value, self.x2.value)

    def check_convexity_conditions(self, relax=False):
        """Returns True if all convexity conditions for the
        conic constraint are satisfied. If relax is True,
        then variable domains are ignored and it is assumed
        that all variables are continuous."""
        return (relax or (self.x1.is_continuous() and self.x2.is_continuous() and self.r.is_continuous())) and (self.x2.has_ub() and value(self.x2.ub) <= 0) and (self.r.has_lb() and value(self.r.lb) >= 0)