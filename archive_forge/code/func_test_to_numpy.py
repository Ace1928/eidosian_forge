from sympy.core.random import randint
from sympy.core.numbers import Integer
from sympy.matrices.dense import (Matrix, ones, zeros)
from sympy.physics.quantum.matrixutils import (
from sympy.external import import_module
from sympy.testing.pytest import skip
def test_to_numpy():
    if not np:
        skip('numpy not installed.')
    result = np.array([[1, 2], [3, 4]], dtype='complex')
    assert (to_numpy(m) == result).all()