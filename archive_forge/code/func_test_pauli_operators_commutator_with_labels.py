from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.matrices.dense import Matrix
from sympy.printing.latex import latex
from sympy.physics.quantum import (Dagger, Commutator, AntiCommutator, qapply,
from sympy.physics.quantum.pauli import (SigmaOpBase, SigmaX, SigmaY, SigmaZ,
from sympy.physics.quantum.pauli import SigmaZKet, SigmaZBra
from sympy.testing.pytest import raises
def test_pauli_operators_commutator_with_labels():
    assert Commutator(sx1, sy1).doit() == 2 * I * sz1
    assert Commutator(sy1, sz1).doit() == 2 * I * sx1
    assert Commutator(sz1, sx1).doit() == 2 * I * sy1
    assert Commutator(sx2, sy2).doit() == 2 * I * sz2
    assert Commutator(sy2, sz2).doit() == 2 * I * sx2
    assert Commutator(sz2, sx2).doit() == 2 * I * sy2
    assert Commutator(sx1, sy2).doit() == 0
    assert Commutator(sy1, sz2).doit() == 0
    assert Commutator(sz1, sx2).doit() == 0