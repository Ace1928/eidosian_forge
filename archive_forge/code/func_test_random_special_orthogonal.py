import numpy as np
import pytest
from cirq.testing import (
from cirq.linalg import is_unitary, is_orthogonal, is_special_unitary, is_special_orthogonal
def test_random_special_orthogonal():
    o1 = random_special_orthogonal(2)
    o2 = random_special_orthogonal(2)
    assert is_special_orthogonal(o1)
    assert is_special_orthogonal(o2)
    assert not np.allclose(o1, o2)