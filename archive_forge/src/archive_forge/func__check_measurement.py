import numpy as np
import pytest
import cirq
import cirq_google
from cirq_google.api import v2
def _check_measurement(m, key, qubits, instances, invert_mask=None, tags=None):
    assert m.key == key
    assert m.qubits == qubits
    assert m.instances == instances
    if invert_mask is not None:
        assert m.invert_mask == invert_mask
    else:
        assert len(m.invert_mask) == len(m.qubits)
        assert m.invert_mask == [False] * len(m.qubits)
    if tags is not None:
        assert m.tags == tags
    else:
        assert len(m.tags) == 0