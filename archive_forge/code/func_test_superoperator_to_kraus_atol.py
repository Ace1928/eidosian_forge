from typing import Iterable, Sequence
import numpy as np
import pytest
import cirq
def test_superoperator_to_kraus_atol():
    """Verifies that insignificant Kraus operators are omitted."""
    superop = cirq.kraus_to_superoperator(cirq.kraus(cirq.phase_damp(1e-06)))
    assert len(cirq.superoperator_to_kraus(superop, atol=0.01)) == 1
    assert len(cirq.superoperator_to_kraus(superop, atol=0.0001)) == 2