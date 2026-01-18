import networkx as nx
import numpy as np
import pytest
import cirq
import cirq.contrib.quimb as ccq
def test_simplify_sandwich():
    rs = np.random.RandomState(52)
    for width in [2, 3]:
        for height in [1, 3]:
            for p in [1, 2]:
                circuit, qubits = _get_circuit(width=width, height=height, p=p, rs=rs)
                operator = cirq.PauliString({q: cirq.Z for q in rs.choice(qubits, size=2, replace=False)})
                tot_c = ccq.circuit_for_expectation_value(circuit, operator)
                tot_c_init = tot_c.copy()
                ccq.simplify_expectation_value_circuit(tot_c)
                assert len(list(tot_c.all_operations())) < len(list(tot_c_init.all_operations()))
                np.testing.assert_allclose(tot_c.unitary(qubit_order=qubits), tot_c_init.unitary(qubit_order=qubits), atol=1e-05)