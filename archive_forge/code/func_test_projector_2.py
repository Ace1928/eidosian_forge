import numpy as np
import pytest
import cirq
def test_projector_2():
    for gate in [cirq.X, cirq.Y, cirq.Z]:
        for eigen_index in [0, 1]:
            eigenvalue = {0: +1, 1: -1}[eigen_index]
            np.testing.assert_allclose(gate.basis[eigenvalue].projector(), gate._eigen_components()[eigen_index][1])