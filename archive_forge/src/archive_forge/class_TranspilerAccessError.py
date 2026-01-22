from qiskit.exceptions import QiskitError
from qiskit.passmanager.exceptions import PassManagerError
class TranspilerAccessError(PassManagerError):
    """DEPRECATED: Exception of access error in the transpiler passes."""