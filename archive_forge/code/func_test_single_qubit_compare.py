import itertools
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra import bit_tools
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('v1,v2', [(v1, v2) for v1 in range(2) for v2 in range(2)])
@allow_deprecated_cirq_ft_use_in_tests
def test_single_qubit_compare(v1: int, v2: int):
    g = cirq_ft.algos.SingleQubitCompare()
    qubits = cirq.LineQid.range(4, dimension=2)
    c = cirq.Circuit(g.on(*qubits))
    initial_state = [v1, v2, 0, 0]
    output_state = [v1, int(v1 == v2), int(v1 < v2), int(v1 > v2)]
    cirq_ft.testing.assert_circuit_inp_out_cirqsim(c, sorted(c.all_qubits()), initial_state, output_state)
    c = cirq.Circuit(g.on(*qubits), (g ** (-1)).on(*qubits))
    cirq_ft.testing.assert_circuit_inp_out_cirqsim(c, sorted(c.all_qubits()), initial_state, initial_state)