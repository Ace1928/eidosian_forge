from cirq.ops import raw_types
class ThreeQubitGate(raw_types.Gate):
    """A gate that must be applied to exactly three qubits."""

    def _num_qubits_(self) -> int:
        return 3