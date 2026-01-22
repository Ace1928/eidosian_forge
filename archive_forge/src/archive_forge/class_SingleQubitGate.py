from cirq.ops import raw_types
class SingleQubitGate(raw_types.Gate):
    """A gate that must be applied to exactly one qubit."""

    def _num_qubits_(self) -> int:
        return 1