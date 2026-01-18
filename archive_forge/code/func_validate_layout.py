from qiskit.circuit import QuantumRegister
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target
@staticmethod
def validate_layout(layout_qubits, dag_qubits):
    """
        Checks if all the qregs in ``layout_qregs`` already exist in ``dag_qregs``. Otherwise, raise.
        """
    for qreg in layout_qubits:
        if qreg not in dag_qubits:
            raise TranspilerError('FullAncillaAllocation: The layout refers to a qubit that does not exist in circuit.')