import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_to_special():
    u = cirq.testing.random_unitary(4)
    su = cirq.to_special(u)
    assert not cirq.is_special_unitary(u)
    assert cirq.is_special_unitary(su)