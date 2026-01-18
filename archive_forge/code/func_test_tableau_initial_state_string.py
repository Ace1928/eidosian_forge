import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('num_qubits', range(1, 4))
def test_tableau_initial_state_string(num_qubits):
    for i in range(2 ** num_qubits):
        t = cirq.CliffordTableau(initial_state=i, num_qubits=num_qubits)
        splitted_represent_string = str(t).split('\n')
        assert len(splitted_represent_string) == num_qubits
        for n in range(num_qubits):
            sign = '- ' if i >> num_qubits - n - 1 & 1 else '+ '
            expected_string = sign + 'I ' * n + 'Z ' + 'I ' * (num_qubits - n - 1)
            assert splitted_represent_string[n] == expected_string