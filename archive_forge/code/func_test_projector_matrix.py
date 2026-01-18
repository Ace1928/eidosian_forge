import numpy as np
import pytest
import cirq
def test_projector_matrix():
    q0 = cirq.NamedQubit('q0')
    zero_projector = cirq.ProjectorString({q0: 0})
    one_projector = cirq.ProjectorString({q0: 1})
    coeff_projector = cirq.ProjectorString({q0: 0}, 1.23 + 4.56j)
    np.testing.assert_allclose(zero_projector.matrix().toarray(), [[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(one_projector.matrix().toarray(), [[0.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(coeff_projector.matrix().toarray(), [[1.23 + 4.56j, 0.0], [0.0, 0.0]])