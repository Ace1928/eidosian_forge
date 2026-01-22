from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
Run the EnlargeWithAncilla pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to extend.

        Returns:
            DAGCircuit: An extended DAG.

        Raises:
            TranspilerError: If there is no layout in the property set or not set at init time.
        