from __future__ import annotations
import warnings
import collections
import numpy as np
import qiskit.circuit.library.standard_gates as gates
from qiskit.circuit import Gate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.utils import optionals
from .gate_sequence import GateSequence
Generates a list of :class:`GateSequence`\ s with the gates in ``basis_gates``.

    Args:
        basis_gates: The gates from which to create the sequences of gates.
        depth: The maximum depth of the approximations.
        filename: If provided, the basic approximations are stored in this file.

    Returns:
        List of :class:`GateSequence`\ s using the gates in ``basis_gates``.

    Raises:
        ValueError: If ``basis_gates`` contains an invalid gate identifier.
    