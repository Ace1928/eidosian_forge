from typing import Optional, Dict, Sequence, Union, cast
import random
import numpy as np
import pytest
import cirq
import cirq.testing
def test_random_circuit_not_expected_number_of_qubits():
    circuit = cirq.testing.random_circuit(qubits=3, n_moments=1, op_density=1.0, gate_domain={cirq.CNOT: 2})
    assert len(circuit.all_qubits()) == 2