from functools import partial
from qiskit.transpiler.passes.optimization.collect_and_collapse import (
from qiskit.quantum_info.operators import Clifford
def _is_clifford_gate(node):
    """Specifies whether a node holds a clifford gate."""
    return node.op.name in clifford_gate_names and getattr(node.op, 'condition', None) is None