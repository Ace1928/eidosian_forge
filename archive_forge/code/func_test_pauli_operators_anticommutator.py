from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.matrices.dense import Matrix
from sympy.printing.latex import latex
from sympy.physics.quantum import (Dagger, Commutator, AntiCommutator, qapply,
from sympy.physics.quantum.pauli import (SigmaOpBase, SigmaX, SigmaY, SigmaZ,
from sympy.physics.quantum.pauli import SigmaZKet, SigmaZBra
from sympy.testing.pytest import raises
def test_pauli_operators_anticommutator():
    assert AntiCommutator(sy, sz).doit() == 0
    assert AntiCommutator(sz, sx).doit() == 0
    assert AntiCommutator(sx, sm).doit() == 1
    assert AntiCommutator(sx, sp).doit() == 1