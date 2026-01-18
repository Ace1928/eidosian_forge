from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.matrices.dense import Matrix
from sympy.printing.latex import latex
from sympy.physics.quantum import (Dagger, Commutator, AntiCommutator, qapply,
from sympy.physics.quantum.pauli import (SigmaOpBase, SigmaX, SigmaY, SigmaZ,
from sympy.physics.quantum.pauli import SigmaZKet, SigmaZBra
from sympy.testing.pytest import raises
def test_pauli_states():
    sx, sz = (SigmaX(), SigmaZ())
    up = SigmaZKet(0)
    down = SigmaZKet(1)
    assert qapply(sx * up) == down
    assert qapply(sx * down) == up
    assert qapply(sz * up) == up
    assert qapply(sz * down) == -down
    up = SigmaZBra(0)
    down = SigmaZBra(1)
    assert qapply(up * sx, dagger=True) == down
    assert qapply(down * sx, dagger=True) == up
    assert qapply(up * sz, dagger=True) == up
    assert qapply(down * sz, dagger=True) == -down
    assert Dagger(SigmaZKet(0)) == SigmaZBra(0)
    assert Dagger(SigmaZBra(1)) == SigmaZKet(1)
    raises(ValueError, lambda: SigmaZBra(2))
    raises(ValueError, lambda: SigmaZKet(2))