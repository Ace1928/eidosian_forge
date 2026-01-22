from typing import Union, Tuple, Sequence, List, Optional
import numpy as np
import cirq
from cirq import ops
from cirq import transformers as opt
def _optimize_multiplexed_angles_circuit(operations: Sequence[ops.Operation]):
    """Removes two qubit gates that amount to identity.
    Exploiting the specific multiplexed structure, this methods looks ahead
    to find stripes of 3 or 4 consecutive CZ or CNOT gates and removes them.

    Args:
        operations: operations to be optimized
    Returns:
        the optimized operations
    """
    circuit = cirq.Circuit(operations)
    circuit = cirq.transformers.drop_negligible_operations(circuit)
    if np.allclose(circuit.unitary(), np.eye(8), atol=1e-14):
        return cirq.Circuit([])

    def num_conseq_2qbit_gates(i):
        j = i
        while j < len(operations) and operations[j].gate.num_qubits() == 2:
            j += 1
        return j - i
    operations = list(circuit.all_operations())
    i = 0
    while i < len(operations):
        num_czs = num_conseq_2qbit_gates(i)
        if num_czs == 4:
            operations = operations[:1]
            break
        elif num_czs == 3:
            operations = operations[:i] + [operations[i + 1]] + operations[i + 3:]
            break
        else:
            i += 1
    return operations