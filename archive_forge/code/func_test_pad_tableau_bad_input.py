import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
def test_pad_tableau_bad_input():
    with pytest.raises(ValueError, match='Input axes of padding should match with the number of qubits'):
        tableau = cirq.CliffordTableau(num_qubits=3)
        cirq.ops.clifford_gate._pad_tableau(tableau, num_qubits_after_padding=4, axes=[1, 2])
    with pytest.raises(ValueError, match='The number of qubits in the input tableau should not be larger than'):
        tableau = cirq.CliffordTableau(num_qubits=3)
        cirq.ops.clifford_gate._pad_tableau(tableau, num_qubits_after_padding=2, axes=[0, 1, 2])