from typing import List
import numpy as np
import pytest
import cirq
def test_respects_nocompile_tags():
    q = cirq.NamedQubit('q')
    c = cirq.Circuit([cirq.Z(q), cirq.H(q), cirq.X(q), cirq.H(q), cirq.X(q).with_tags('nocompile'), cirq.H(q)])
    context = cirq.TransformerContext(tags_to_ignore=('nocompile',))
    c = cirq.drop_empty_moments(cirq.merge_k_qubit_unitaries(c, k=1, context=context))
    assert len(c) == 3
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(c[0]), np.eye(2), atol=1e-07)
    assert c[1][q] == cirq.X(q).with_tags('nocompile')
    assert isinstance(c[-1][q].gate, cirq.MatrixGate)