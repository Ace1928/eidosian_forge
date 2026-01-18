import numpy as np
import pytest
import cirq
def test_projector_matrix_missing_qid():
    q0, q1 = cirq.LineQubit.range(2)
    proj = cirq.ProjectorString({q0: 0})
    proj_with_coefficient = cirq.ProjectorString({q0: 0}, 1.23 + 4.56j)
    np.testing.assert_allclose(proj.matrix().toarray(), np.diag([1.0, 0.0]))
    np.testing.assert_allclose(proj.matrix([q0]).toarray(), np.diag([1.0, 0.0]))
    np.testing.assert_allclose(proj.matrix([q1]).toarray(), np.diag([1.0, 1.0]))
    np.testing.assert_allclose(proj.matrix([q0, q1]).toarray(), np.diag([1.0, 1.0, 0.0, 0.0]))
    np.testing.assert_allclose(proj.matrix([q1, q0]).toarray(), np.diag([1.0, 0.0, 1.0, 0.0]))
    np.testing.assert_allclose(proj_with_coefficient.matrix([q1, q0]).toarray(), np.diag([1.23 + 4.56j, 0.0, 1.23 + 4.56j, 0.0]))