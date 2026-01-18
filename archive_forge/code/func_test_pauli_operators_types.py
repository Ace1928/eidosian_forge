from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.matrices.dense import Matrix
from sympy.printing.latex import latex
from sympy.physics.quantum import (Dagger, Commutator, AntiCommutator, qapply,
from sympy.physics.quantum.pauli import (SigmaOpBase, SigmaX, SigmaY, SigmaZ,
from sympy.physics.quantum.pauli import SigmaZKet, SigmaZBra
from sympy.testing.pytest import raises
def test_pauli_operators_types():
    assert isinstance(sx, SigmaOpBase) and isinstance(sx, SigmaX)
    assert isinstance(sy, SigmaOpBase) and isinstance(sy, SigmaY)
    assert isinstance(sz, SigmaOpBase) and isinstance(sz, SigmaZ)
    assert isinstance(sm, SigmaOpBase) and isinstance(sm, SigmaMinus)
    assert isinstance(sp, SigmaOpBase) and isinstance(sp, SigmaPlus)