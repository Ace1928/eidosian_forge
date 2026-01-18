from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.exceptions import QiskitError
from qiskit.circuit import ControlFlowOp
from qiskit.converters.circuit_to_dag import circuit_to_dag
Run the Unroll3qOrMore pass on `dag`.

        Args:
            dag(DAGCircuit): input dag
        Returns:
            DAGCircuit: output dag with maximum node degrees of 2
        Raises:
            QiskitError: if a 3q+ gate is not decomposable
        