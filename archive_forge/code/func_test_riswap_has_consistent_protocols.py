import numpy as np
import pytest
import sympy
from scipy import linalg
import cirq
@pytest.mark.parametrize('angle_rads', (-np.pi / 5, 0.4, 2, np.pi))
def test_riswap_has_consistent_protocols(angle_rads):
    cirq.testing.assert_implements_consistent_protocols(cirq.riswap(angle_rads))