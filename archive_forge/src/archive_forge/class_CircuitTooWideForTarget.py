from qiskit.exceptions import QiskitError
from qiskit.passmanager.exceptions import PassManagerError
class CircuitTooWideForTarget(TranspilerError):
    """Error raised if the circuit is too wide for the target."""