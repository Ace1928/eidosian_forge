from abc import ABC, abstractmethod
from typing import Any, Dict
class EvolutionSynthesis(ABC):
    """Interface for evolution synthesis algorithms."""

    @abstractmethod
    def synthesize(self, evolution):
        """Synthesize an ``qiskit.circuit.library.PauliEvolutionGate``.

        Args:
            evolution (PauliEvolutionGate): The evolution gate to synthesize.

        Returns:
            QuantumCircuit: A circuit implementing the evolution.
        """
        raise NotImplementedError

    @property
    def settings(self) -> Dict[str, Any]:
        """Return the settings in a dictionary, which can be used to reconstruct the object.

        Returns:
            A dictionary containing the settings of this product formula.

        Raises:
            NotImplementedError: The interface does not implement this method.
        """
        raise NotImplementedError('The settings property is not implemented for the base interface.')