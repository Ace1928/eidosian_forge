from typing import Optional
from abc import ABC, abstractmethod
from ..blueprintcircuit import BlueprintCircuit
class FunctionalPauliRotations(BlueprintCircuit, ABC):
    """Base class for functional Pauli rotations."""

    def __init__(self, num_state_qubits: Optional[int]=None, basis: str='Y', name: str='F') -> None:
        """Create a new functional Pauli rotation circuit.

        Args:
            num_state_qubits: The number of qubits representing the state :math:`|x\\rangle`.
            basis: The kind of Pauli rotation to use. Must be 'X', 'Y' or 'Z'.
            name: The name of the circuit object.
        """
        super().__init__(name=name)
        self._num_state_qubits = None
        self._basis = None
        self.num_state_qubits = num_state_qubits
        self.basis = basis

    @property
    def basis(self) -> str:
        """The kind of Pauli rotation to be used.

        Set the basis to 'X', 'Y' or 'Z' for controlled-X, -Y, or -Z rotations respectively.

        Returns:
            The kind of Pauli rotation used in controlled rotation.
        """
        return self._basis

    @basis.setter
    def basis(self, basis: str) -> None:
        """Set the kind of Pauli rotation to be used.

        Args:
            basis: The Pauli rotation to be used.

        Raises:
            ValueError: The provided basis in not X, Y or Z.
        """
        basis = basis.lower()
        if self._basis is None or basis != self._basis:
            if basis not in ['x', 'y', 'z']:
                raise ValueError(f'The provided basis must be X, Y or Z, not {basis}')
            self._invalidate()
            self._basis = basis

    @property
    def num_state_qubits(self) -> int:
        """The number of state qubits representing the state :math:`|x\\rangle`.

        Returns:
            The number of state qubits.
        """
        return self._num_state_qubits

    @num_state_qubits.setter
    def num_state_qubits(self, num_state_qubits: Optional[int]) -> None:
        """Set the number of state qubits.

        Note that this may change the underlying quantum register, if the number of state qubits
        changes.

        Args:
            num_state_qubits: The new number of qubits.
        """
        if self._num_state_qubits is None or num_state_qubits != self._num_state_qubits:
            self._invalidate()
            self._num_state_qubits = num_state_qubits
            self._reset_registers(num_state_qubits)

    @abstractmethod
    def _reset_registers(self, num_state_qubits: Optional[int]) -> None:
        """Reset the registers according to the new number of state qubits.

        Args:
            num_state_qubits: The new number of qubits.
        """
        raise NotImplementedError

    @property
    def num_ancilla_qubits(self) -> int:
        """The minimum number of ancilla qubits in the circuit.

        Returns:
            The minimal number of ancillas required.
        """
        return 0