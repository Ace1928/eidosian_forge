import abc
from typing import List
import stevedore
class HighLevelSynthesisPlugin(abc.ABC):
    """Abstract high-level synthesis plugin class.

    This abstract class defines the interface for high-level synthesis plugins.
    """

    @abc.abstractmethod
    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Operation.

        Args:
            high_level_object (Operation): The Operation to synthesize to a
                :class:`~qiskit.dagcircuit.DAGCircuit` object.
            coupling_map (CouplingMap): The coupling map of the backend
                in case synthesis is done on a physical circuit.
            target (Target): A target representing the target backend.
            qubits (list): List of qubits over which the operation is defined
                in case synthesis is done on a physical circuit.
            options: Additional method-specific optional kwargs.

        Returns:
            QuantumCircuit: The quantum circuit representation of the Operation
                when successful, and ``None`` otherwise.
        """
        pass