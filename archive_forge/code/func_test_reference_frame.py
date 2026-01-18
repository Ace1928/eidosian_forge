from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import (eye, zeros)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.simplify.simplify import simplify
from sympy.physics.vector import (ReferenceFrame, Vector, CoordinateSym,
from sympy.physics.vector.frame import _check_frame
from sympy.physics.vector.vector import VectorTypeError
from sympy.testing.pytest import raises
import warnings
def test_reference_frame():
    raises(TypeError, lambda: ReferenceFrame(0))
    raises(TypeError, lambda: ReferenceFrame('N', 0))
    raises(ValueError, lambda: ReferenceFrame('N', [0, 1]))
    raises(TypeError, lambda: ReferenceFrame('N', [0, 1, 2]))
    raises(TypeError, lambda: ReferenceFrame('N', ['a', 'b', 'c'], 0))
    raises(ValueError, lambda: ReferenceFrame('N', ['a', 'b', 'c'], [0, 1]))
    raises(TypeError, lambda: ReferenceFrame('N', ['a', 'b', 'c'], [0, 1, 2]))
    raises(TypeError, lambda: ReferenceFrame('N', ['a', 'b', 'c'], ['a', 'b', 'c'], 0))
    raises(ValueError, lambda: ReferenceFrame('N', ['a', 'b', 'c'], ['a', 'b', 'c'], [0, 1]))
    raises(TypeError, lambda: ReferenceFrame('N', ['a', 'b', 'c'], ['a', 'b', 'c'], [0, 1, 2]))
    N = ReferenceFrame('N')
    assert N[0] == CoordinateSym('N_x', N, 0)
    assert N[1] == CoordinateSym('N_y', N, 1)
    assert N[2] == CoordinateSym('N_z', N, 2)
    raises(ValueError, lambda: N[3])
    N = ReferenceFrame('N', ['a', 'b', 'c'])
    assert N['a'] == N.x
    assert N['b'] == N.y
    assert N['c'] == N.z
    raises(ValueError, lambda: N['d'])
    assert str(N) == 'N'
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
    raises(TypeError, lambda: A.orient(B, 'DCM', 0))
    raises(TypeError, lambda: B.orient(N, 'Space', [q1, q2, q3], '222'))
    raises(TypeError, lambda: B.orient(N, 'Axis', [q1, N.x + 2 * N.y], '222'))
    raises(TypeError, lambda: B.orient(N, 'Axis', q1))
    raises(IndexError, lambda: B.orient(N, 'Axis', [q1]))
    raises(TypeError, lambda: B.orient(N, 'Quaternion', [q0, q1, q2, q3], '222'))
    raises(TypeError, lambda: B.orient(N, 'Quaternion', q0))
    raises(TypeError, lambda: B.orient(N, 'Quaternion', [q0, q1, q2]))
    raises(NotImplementedError, lambda: B.orient(N, 'Foo', [q0, q1, q2]))
    raises(TypeError, lambda: B.orient(N, 'Body', [q1, q2], '232'))
    raises(TypeError, lambda: B.orient(N, 'Space', [q1, q2], '232'))
    N.set_ang_acc(B, 0)
    assert N.ang_acc_in(B) == Vector(0)
    N.set_ang_vel(B, 0)
    assert N.ang_vel_in(B) == Vector(0)