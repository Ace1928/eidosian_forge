from sympy.core.numbers import (I, Rational, pi)
from sympy.core.power import Pow
from sympy.functions.elementary.exponential import exp
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.matrixutils import (
def _scipy_sparse_matrix(self, name, m):
    m = to_scipy_sparse(m, dtype=self.dtype)
    self._store_matrix(name, 'scipy.sparse', m)