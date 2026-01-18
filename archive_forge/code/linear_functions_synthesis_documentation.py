from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.circuit.library import PermutationGate
from qiskit.circuit.exceptions import CircuitError
Run the LinearFunctionsToPermutations pass on `dag`.
        Args:
            dag: input dag.
        Returns:
            Output dag with LinearFunctions synthesized.
        