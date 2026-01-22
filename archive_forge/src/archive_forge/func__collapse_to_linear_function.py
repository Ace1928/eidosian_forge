from functools import partial
from qiskit.circuit.library.generalized_gates import LinearFunction
from qiskit.transpiler.passes.optimization.collect_and_collapse import (
def _collapse_to_linear_function(circuit):
    """Specifies how to construct a ``LinearFunction`` from a quantum circuit (that must
    consist of linear gates only)."""
    return LinearFunction(circuit)