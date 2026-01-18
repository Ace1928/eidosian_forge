import numpy as np
import pytest
import cirq
def test_projector_from_np_array():
    q0 = cirq.NamedQubit('q0')
    zero_projector = cirq.ProjectorString({q0: 0})
    np.testing.assert_allclose(zero_projector.matrix().toarray(), [[1.0, 0.0], [0.0, 0.0]])