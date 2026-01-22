from qiskit.circuit.annotated_operation import AnnotatedOperation
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.exceptions import QiskitError
def _check_is_gate(op):
    """Checks whether op can be converted to Gate."""
    if isinstance(op, Gate):
        return True
    elif isinstance(op, AnnotatedOperation):
        return _check_is_gate(op.base_op)
    return False