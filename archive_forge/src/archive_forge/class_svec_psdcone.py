from pyomo.core.expr.numvalue import is_numeric_data
from pyomo.core.expr import value, exp
from pyomo.core.kernel.block import block
from pyomo.core.kernel.variable import IVariable, variable, variable_tuple
from pyomo.core.kernel.constraint import (
class svec_psdcone(_ConicBase):
    """A domain consisting of vectorizations of the lower-triangular
    part of a positive semidefinite matrx, with the non-diagonal
    elements additionally rescaled. In other words, if a vector 'x'
    of length n = d*(d+1)/2 belongs to this cone, then the matrix:

    sMat(x) = [[        x[1],    x[2]/sqrt(2),  ...,         x[d]/sqrt(2)],
               [x[2]/sqrt(2),          x[d+1],  ...,      x[2d-1]/sqrt(2)],
                                        ...
               [x[d]/sqrt(2), x[2d-1]/sqrt(2),  ..., x[d*(d+1)/2]/sqrt(2)]]

    will be restricted to be a positive-semidefinite matrix.

    Parameters
    ----------
    x : :class:`variable`
        An iterable of variables with length d*(d+1)/2.

    """
    __slots__ = ('_parent', '_storage_key', '_active', '_body', '_x', '__weakref__')

    def __init__(self, x):
        super(svec_psdcone, self).__init__()
        self._x = tuple(x)
        assert all((isinstance(xi, IVariable) for xi in self._x))

    @classmethod
    def as_domain(cls, x):
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
        b.x = variable_tuple([variable() for i in range(len(x))])
        b.c = _build_linking_constraints(list(x), list(b.x))
        b.q = cls(x=b.x)
        return b

    @property
    def x(self):
        return self._x