from __future__ import annotations
from collections import defaultdict
from typing import Literal
import numpy as np
import rustworkx as rx
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.custom_iterator import CustomIterator
from qiskit.quantum_info.operators.mixins import GroupMixin, LinearMixin
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli
from qiskit.quantum_info.operators.symplectic.clifford import Clifford
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
def _commutes_with_all(self, other, anti=False):
    """Return row indexes that commute with all rows in another PauliList.

        Args:
            other (PauliList): a PauliList.
            anti (bool): if ``True`` return rows that anti-commute, otherwise
                         return rows that commute (Default: ``False``).

        Returns:
            array: index array of commuting or anti-commuting row.
        """
    if not isinstance(other, PauliList):
        other = PauliList(other)
    comms = self.commutes(other[0])
    inds, = np.where(comms == int(not anti))
    for pauli in other[1:]:
        comms = self[inds].commutes(pauli)
        new_inds, = np.where(comms == int(not anti))
        if new_inds.size == 0:
            return new_inds
        inds = inds[new_inds]
    return inds