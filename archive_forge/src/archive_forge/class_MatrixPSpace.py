from math import prod
from sympy.core.basic import Basic
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.special.gamma_functions import multigamma
from sympy.core.sympify import sympify, _sympify
from sympy.matrices import (ImmutableMatrix, Inverse, Trace, Determinant,
from sympy.stats.rv import (_value_check, RandomMatrixSymbol, NamedArgsMixin, PSpace,
from sympy.external import import_module
class MatrixPSpace(PSpace):
    """
    Represents probability space for
    Matrix Distributions.
    """

    def __new__(cls, sym, distribution, dim_n, dim_m):
        sym = _symbol_converter(sym)
        dim_n, dim_m = (_sympify(dim_n), _sympify(dim_m))
        if not (dim_n.is_integer and dim_m.is_integer):
            raise ValueError('Dimensions should be integers')
        return Basic.__new__(cls, sym, distribution, dim_n, dim_m)
    distribution = property(lambda self: self.args[1])
    symbol = property(lambda self: self.args[0])

    @property
    def domain(self):
        return MatrixDomain(self.symbol, self.distribution.set)

    @property
    def value(self):
        return RandomMatrixSymbol(self.symbol, self.args[2], self.args[3], self)

    @property
    def values(self):
        return {self.value}

    def compute_density(self, expr, *args):
        rms = expr.atoms(RandomMatrixSymbol)
        if len(rms) > 1 or not isinstance(expr, RandomMatrixSymbol):
            raise NotImplementedError('Currently, no algorithm has been implemented to handle general expressions containing multiple matrix distributions.')
        return self.distribution.pdf(expr)

    def sample(self, size=(), library='scipy', seed=None):
        """
        Internal sample method

        Returns dictionary mapping RandomMatrixSymbol to realization value.
        """
        return {self.value: self.distribution.sample(size, library=library, seed=seed)}