import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_stabilizerStateChForm_H():
    q0, q1 = (cirq.LineQubit(0), cirq.LineQubit(1))
    state = cirq.CliffordState(qubit_map={q0: 0, q1: 1})
    with pytest.raises(ValueError, match='|y> is equal to |z>'):
        state.ch_form._H_decompose(0, 1, 1, 0)