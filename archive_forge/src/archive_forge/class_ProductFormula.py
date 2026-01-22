from typing import Callable, Optional, Union, Any, Dict
from functools import partial
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli
from .evolution_synthesis import EvolutionSynthesis
class ProductFormula(EvolutionSynthesis):
    """Product formula base class for the decomposition of non-commuting operator exponentials.

    :obj:`.LieTrotter` and :obj:`.SuzukiTrotter` inherit from this class.
    """

    def __init__(self, order: int, reps: int=1, insert_barriers: bool=False, cx_structure: str='chain', atomic_evolution: Optional[Callable[[Union[Pauli, SparsePauliOp], float], QuantumCircuit]]=None) -> None:
        """
        Args:
            order: The order of the product formula.
            reps: The number of time steps.
            insert_barriers: Whether to insert barriers between the atomic evolutions.
            cx_structure: How to arrange the CX gates for the Pauli evolutions, can be
                ``"chain"``, where next neighbor connections are used, or ``"fountain"``,
                where all qubits are connected to one.
            atomic_evolution: A function to construct the circuit for the evolution of single
                Pauli string. Per default, a single Pauli evolution is decomposed in a CX chain
                and a single qubit Z rotation.
        """
        super().__init__()
        self.order = order
        self.reps = reps
        self.insert_barriers = insert_barriers
        self._atomic_evolution = atomic_evolution
        self._cx_structure = cx_structure
        if atomic_evolution is None:
            atomic_evolution = partial(_default_atomic_evolution, cx_structure=cx_structure)
        self.atomic_evolution = atomic_evolution

    @property
    def settings(self) -> Dict[str, Any]:
        """Return the settings in a dictionary, which can be used to reconstruct the object.

        Returns:
            A dictionary containing the settings of this product formula.

        Raises:
            NotImplementedError: If a custom atomic evolution is set, which cannot be serialized.
        """
        if self._atomic_evolution is not None:
            raise NotImplementedError('Cannot serialize a product formula with a custom atomic evolution.')
        return {'order': self.order, 'reps': self.reps, 'insert_barriers': self.insert_barriers, 'cx_structure': self._cx_structure}