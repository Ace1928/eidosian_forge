from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer)
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import conjugate
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.dagger import adjoint, Dagger
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.physics.quantum.operator import Operator, IdentityOperator
def test_scipy_sparse_dagger():
    if not np:
        skip('numpy not installed.')
    if not scipy:
        skip('scipy not installed.')
    else:
        sparse = scipy.sparse
    a = sparse.csr_matrix([[1.0 + 0j, 2j], [-1j, 2.0 + 0j]])
    adag = a.copy().transpose().conjugate()
    assert np.linalg.norm((Dagger(a) - adag).todense()) == 0.0