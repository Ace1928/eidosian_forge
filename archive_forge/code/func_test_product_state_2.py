import numpy as np
import pytest
import cirq
def test_product_state_2():
    q0, q1 = cirq.LineQubit.range(2)
    with pytest.raises(ValueError):
        _ = cirq.KET_PLUS(q0) * cirq.KET_PLUS(q1) * -1
    with pytest.raises(ValueError):
        _ = cirq.KET_PLUS(q0) * cirq.KET_PLUS(q1) * cirq.KET_ZERO