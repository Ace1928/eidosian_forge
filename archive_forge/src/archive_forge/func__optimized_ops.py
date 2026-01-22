from typing import Callable
from cirq import ops, circuits, transformers
from cirq.contrib.paulistring.pauli_string_optimize import pauli_string_optimized_circuit
from cirq.contrib.paulistring.clifford_optimize import clifford_optimized_circuit
def _optimized_ops(ops: ops.OP_TREE, atol: float=1e-08, repeat: int=10) -> ops.OP_TREE:
    c = circuits.Circuit(ops)
    c_opt = optimized_circuit(c, atol, repeat, merge_interactions=False)
    return [*c_opt.all_operations()]