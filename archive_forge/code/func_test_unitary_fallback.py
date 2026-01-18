from typing import Tuple
import numpy as np
import pytest
import cirq
def test_unitary_fallback():

    class UnitaryXGate(cirq.testing.SingleQubitGate):

        def _unitary_(self):
            return np.array([[0, 1], [1, 0]])

    class UnitaryYGate(cirq.Gate):

        def _qid_shape_(self) -> Tuple[int, ...]:
            return (2,)

        def _unitary_(self):
            return np.array([[0, -1j], [1j, 0]])
    original_tableau = cirq.CliffordTableau(num_qubits=3)
    args = cirq.CliffordTableauSimulationState(tableau=original_tableau.copy(), qubits=cirq.LineQubit.range(3), prng=np.random.RandomState())
    cirq.act_on(UnitaryXGate(), args, [cirq.LineQubit(1)])
    assert args.tableau == cirq.CliffordTableau(num_qubits=3, initial_state=2)
    args = cirq.CliffordTableauSimulationState(tableau=original_tableau.copy(), qubits=cirq.LineQubit.range(3), prng=np.random.RandomState())
    cirq.act_on(UnitaryYGate(), args, [cirq.LineQubit(1)])
    expected_args = cirq.CliffordTableauSimulationState(tableau=original_tableau.copy(), qubits=cirq.LineQubit.range(3), prng=np.random.RandomState())
    cirq.act_on(cirq.Y, expected_args, [cirq.LineQubit(1)])
    assert args.tableau == expected_args.tableau