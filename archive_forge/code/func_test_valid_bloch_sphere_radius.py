import math
import cirq
import pytest
import numpy as np
import cirq_web
@pytest.mark.parametrize('sphere_radius', [5, 0.2, 100])
def test_valid_bloch_sphere_radius(sphere_radius):
    bloch_sphere = cirq_web.BlochSphere(sphere_radius, [1 / math.sqrt(2), 1 / math.sqrt(2)])
    assert sphere_radius == bloch_sphere.sphere_radius