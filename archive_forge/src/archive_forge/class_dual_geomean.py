from pyomo.core.expr.numvalue import is_numeric_data
from pyomo.core.expr import value, exp
from pyomo.core.kernel.block import block
from pyomo.core.kernel.variable import IVariable, variable, variable_tuple
from pyomo.core.kernel.constraint import (
class dual_geomean(_ConicBase):
    """A dual geometric mean conic constraint of the form:
        (n-1)*(r[0]*...*r[n-2])^(1/(n-1)) >= |x[n-1]|

    Parameters
    ----------
    r : :class:`variable`
        An iterable of variables.
    x : :class:`variable`
        A scalar variable.

    """
    __slots__ = ('_parent', '_storage_key', '_active', '_body', '_r', '_x', '__weakref__')

    def __init__(self, r, x):
        super(dual_geomean, self).__init__()
        self._r = tuple(r)
        self._x = x
        assert isinstance(self._x, IVariable)
        assert all((isinstance(ri, IVariable) for ri in self._r))

    @classmethod
    def as_domain(cls, r, x):
        """Builds a conic domain. Input arguments take the
        same form as those of the conic constraint, but in
        place of each variable, one can optionally supply a
        constant, linear expression, or None.

        Returns
        -------
        block
            A block object with the core conic constraint
            (block.q) expressed using auxiliary variables
            (block.r, block.x) linked to the input arguments
            through auxiliary constraints (block.c)."""
        b = block()
        b.r = variable_tuple([variable(lb=0) for i in range(len(r))])
        b.x = variable()
        b.c = _build_linking_constraints(list(r) + [x], list(b.r) + [x])
        b.q = cls(r=b.r, x=b.x)
        return b

    @property
    def r(self):
        return self._r

    @property
    def x(self):
        return self._x