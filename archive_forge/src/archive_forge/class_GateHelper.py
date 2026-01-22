from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
from numpy.typing import NDArray
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft.infra import gate_with_registers, t_complexity_protocol, merge_qubits, get_named_qubits
from cirq_ft.infra.decompose_protocol import _decompose_once_considering_known_decomposition
@dataclass(frozen=True)
class GateHelper:
    """A collection of related objects derivable from a `GateWithRegisters`.

    These are likely useful to have at one's fingertips while writing tests or
    demo notebooks.

    Attributes:
        gate: The gate from which all other objects are derived.
    """
    gate: gate_with_registers.GateWithRegisters
    context: cirq.DecompositionContext = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())

    @cached_property
    def r(self) -> gate_with_registers.Signature:
        """The Signature system for the gate."""
        return self.gate.signature

    @cached_property
    def quregs(self) -> Dict[str, NDArray[cirq.Qid]]:
        """A dictionary of named qubits appropriate for the signature for the gate."""
        return get_named_qubits(self.r)

    @cached_property
    def all_qubits(self) -> List[cirq.Qid]:
        """All qubits in Register order."""
        merged_qubits = merge_qubits(self.r, **self.quregs)
        decomposed_qubits = self.decomposed_circuit.all_qubits()
        return merged_qubits + sorted(decomposed_qubits - frozenset(merged_qubits))

    @cached_property
    def operation(self) -> cirq.Operation:
        """The `gate` applied to example qubits."""
        return self.gate.on_registers(**self.quregs)

    @cached_property
    def circuit(self) -> cirq.Circuit:
        """The `gate` applied to example qubits wrapped in a `cirq.Circuit`."""
        return cirq.Circuit(self.operation)

    @cached_property
    def decomposed_circuit(self) -> cirq.Circuit:
        """The `gate` applied to example qubits, decomposed and wrapped in a `cirq.Circuit`."""
        return cirq.Circuit(cirq.decompose(self.operation, context=self.context))