import numpy as np
import pytest
import cirq
import cirq_google as cg
def test_syc_unitary():
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(cg.SYC), np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1j, 0.0], [0.0, -1j, 0.0, 0.0], [0.0, 0.0, 0.0, np.exp(-1j * np.pi / 6)]]), atol=1e-06)