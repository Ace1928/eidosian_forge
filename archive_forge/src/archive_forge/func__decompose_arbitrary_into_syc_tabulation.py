from typing import Iterator, List, Optional
import itertools
import math
import numpy as np
import cirq
from cirq_google import ops
def _decompose_arbitrary_into_syc_tabulation(op: cirq.Operation, tabulation: cirq.TwoQubitGateTabulation) -> cirq.OP_TREE:
    """Synthesize an arbitrary 2 qubit operation to a Sycamore operation using the given Tabulation.

    Args:
        op: Operation to decompose.
        tabulation: A `cirq.TwoQubitGateTabulation` for the Sycamore gate.

    Yields:
        A `cirq.OP_TREE` that performs the given operation using Sycamore operations.
    """
    qubit_a, qubit_b = op.qubits
    result = tabulation.compile_two_qubit_gate(cirq.unitary(op))
    local_gates = result.local_unitaries
    for i, (gate_a, gate_b) in enumerate(local_gates):
        yield from _phased_x_z_ops(gate_a, qubit_a)
        yield from _phased_x_z_ops(gate_b, qubit_b)
        if i != len(local_gates) - 1:
            yield ops.SYC.on(qubit_a, qubit_b)