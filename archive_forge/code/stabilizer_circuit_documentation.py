from __future__ import annotations
from collections.abc import Collection
import numpy as np
from qiskit.quantum_info import PauliList
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.operators.symplectic.clifford import Clifford
Synthesis of a circuit that generates a state stabilized by the stabilizers
    using Gaussian elimination with Clifford gates.
    If the stabilizers are underconstrained, and ``allow_underconstrained`` is ``True``,
    the circuit will output one of the states stabilized by the stabilizers.
    Based on stim implementation.

    Args:
        stabilizers: List of stabilizer strings
        allow_redundant: Allow redundant stabilizers (i.e., some stabilizers
            can be products of the others)
        allow_underconstrained: Allow underconstrained set of stabilizers (i.e.,
            the stabilizers do not specify a unique state)
        invert: Return inverse circuit

    Returns:
        A circuit that generates a state stabilized by ``stabilizers``.

    Raises:
        QiskitError: if the stabilizers are invalid, do not commute, or contradict each other,
                     if the list is underconstrained and ``allow_underconstrained`` is ``False``,
                     or if the list is redundant and ``allow_redundant`` is ``False``.

    References:
        1. https://github.com/quantumlib/Stim/blob/c0dd0b1c8125b2096cd54b6f72884a459e47fe3e/src/stim/stabilizers/conversions.inl#L469
        2. https://quantumcomputing.stackexchange.com/questions/12721/how-to-calculate-destabilizer-group-of-toric-and-other-codes

    