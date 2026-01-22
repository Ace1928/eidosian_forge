from typing import Any, Dict, Optional, Sequence
import cirq
class EmptySimulationState(cirq.SimulationState):

    def __init__(self, qubits, classical_data):
        super().__init__(state=EmptyQuantumState(), qubits=qubits, classical_data=classical_data)

    def _act_on_fallback_(self, action: Any, qubits: Sequence['cirq.Qid'], allow_decompose: bool=True) -> bool:
        return True