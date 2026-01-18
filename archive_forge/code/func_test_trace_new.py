from sympy.core.containers import Tuple
from sympy.core.symbol import symbols
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.trace import Tr
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_trace_new():
    a, b, c, d, Y = symbols('a b c d Y')
    A, B, C, D = symbols('A B C D', commutative=False)
    assert Tr(a + b) == a + b
    assert Tr(A + B) == Tr(A) + Tr(B)
    assert Tr(C * D * A * B).args[0].args == (C, D, A, B)
    assert Tr(a * b + c * d) == a * b + c * d
    assert Tr(a * A) == a * Tr(A)
    assert Tr(a * A * B * b) == a * b * Tr(A * B)
    assert isinstance(Tr(A), Tr)
    assert Tr(pow(a, b)) == a ** b
    assert isinstance(Tr(pow(A, a)), Tr)
    M = Matrix([[1, 1], [2, 2]])
    assert Tr(M) == 3
    t = Tr(A)
    assert t.args[1] == Tuple()
    t = Tr(A, 0)
    assert t.args[1] == Tuple(0)
    t = Tr(A, [0])
    assert t.args[1] == Tuple(0)
    t = Tr(A, [0, 1, 2])
    assert t.args[1] == Tuple(0, 1, 2)
    t = Tr(A, 0)
    assert t.args[1] == Tuple(0)
    t = Tr(A, (1, 2))
    assert t.args[1] == Tuple(1, 2)
    t = Tr(A + B, [2])
    assert t.args[0].args[1] == Tuple(2) and t.args[1].args[1] == Tuple(2)
    t = Tr(a * A, [2, 3])
    assert t.args[1].args[1] == Tuple(2, 3)

    class Foo:

        def trace(self):
            return 1
    assert Tr(Foo()) == 1
    raises(ValueError, lambda: Tr())
    raises(ValueError, lambda: Tr(A, 1, 2))