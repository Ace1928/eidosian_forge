from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational, pi)
from sympy.core.symbol import (Wild, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices import Matrix, ImmutableMatrix
from sympy.physics.quantum.gate import (XGate, YGate, ZGate, random_circuit,
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qubit import Qubit, IntQubit, qubit_to_matrix, \
from sympy.physics.quantum.matrixutils import matrix_to_zero
from sympy.physics.quantum.matrixcache import sqrt2_inv
from sympy.physics.quantum import Dagger
def test_one_qubit_commutators():
    """Test single qubit gate commutation relations."""
    for g1 in (IdentityGate, X, Y, Z, H, T, S):
        for g2 in (IdentityGate, X, Y, Z, H, T, S):
            e = Commutator(g1(0), g2(0))
            a = matrix_to_zero(represent(e, nqubits=1, format='sympy'))
            b = matrix_to_zero(represent(e.doit(), nqubits=1, format='sympy'))
            assert a == b
            e = Commutator(g1(0), g2(1))
            assert e.doit() == 0