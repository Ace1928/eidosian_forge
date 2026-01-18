from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.matrices.dense import Matrix
from sympy.printing.latex import latex
from sympy.physics.quantum import (Dagger, Commutator, AntiCommutator, qapply,
from sympy.physics.quantum.pauli import (SigmaOpBase, SigmaX, SigmaY, SigmaZ,
from sympy.physics.quantum.pauli import SigmaZKet, SigmaZBra
from sympy.testing.pytest import raises
def test_pauli_operators_multiplication():
    assert qsimplify_pauli(sx * sx) == 1
    assert qsimplify_pauli(sy * sy) == 1
    assert qsimplify_pauli(sz * sz) == 1
    assert qsimplify_pauli(sx * sy) == I * sz
    assert qsimplify_pauli(sy * sz) == I * sx
    assert qsimplify_pauli(sz * sx) == I * sy
    assert qsimplify_pauli(sy * sx) == -I * sz
    assert qsimplify_pauli(sz * sy) == -I * sx
    assert qsimplify_pauli(sx * sz) == -I * sy