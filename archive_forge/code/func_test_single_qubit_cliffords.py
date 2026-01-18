import numpy as np
import pytest
import matplotlib.pyplot as plt
import cirq
import cirq.experiments.qubit_characterizations as ceqc
from cirq import GridQubit
from cirq import circuits, ops, sim
from cirq.experiments import (
def test_single_qubit_cliffords():
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1, -1])
    PAULIS = (I, X, Y, Z)

    def is_pauli(u):
        return any((cirq.equal_up_to_global_phase(u, p) for p in PAULIS))
    cliffords = ceqc._single_qubit_cliffords()
    assert len(cliffords.c1_in_xy) == 24
    assert len(cliffords.c1_in_xz) == 24

    def unitary(gates):
        U = np.eye(2)
        for gate in gates:
            U = cirq.unitary(gate) @ U
        return U
    xy_unitaries = [unitary(gates) for gates in cliffords.c1_in_xy]
    xz_unitaries = [unitary(gates) for gates in cliffords.c1_in_xz]

    def check_distinct(unitaries):
        n = len(unitaries)
        for i in range(n):
            for j in range(i + 1, n):
                Ui, Uj = (unitaries[i], unitaries[j])
                assert not cirq.allclose_up_to_global_phase(Ui, Uj), f'{i}, {j}'
    check_distinct(xy_unitaries)
    check_distinct(xz_unitaries)
    for Uxy in xy_unitaries:
        assert any((cirq.allclose_up_to_global_phase(Uxy, Uxz) for Uxz in xz_unitaries))
    for u in xy_unitaries:
        for p in PAULIS:
            assert is_pauli(u @ p @ u.conj().T), str(u)
    for gates in cliffords.c1_in_xz:
        num_x = len([gate for gate in gates if isinstance(gate, cirq.XPowGate)])
        num_z = len([gate for gate in gates if isinstance(gate, cirq.ZPowGate)])
        assert num_x + num_z == len(gates)
        assert num_x <= 1