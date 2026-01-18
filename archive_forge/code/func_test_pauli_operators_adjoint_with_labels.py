from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.matrices.dense import Matrix
from sympy.printing.latex import latex
from sympy.physics.quantum import (Dagger, Commutator, AntiCommutator, qapply,
from sympy.physics.quantum.pauli import (SigmaOpBase, SigmaX, SigmaY, SigmaZ,
from sympy.physics.quantum.pauli import SigmaZKet, SigmaZBra
from sympy.testing.pytest import raises
def test_pauli_operators_adjoint_with_labels():
    assert Dagger(sx1) == sx1
    assert Dagger(sy1) == sy1
    assert Dagger(sz1) == sz1
    assert Dagger(sx1) != sx2
    assert Dagger(sy1) != sy2
    assert Dagger(sz1) != sz2