from __future__ import annotations
import warnings
import collections
import numpy as np
import qiskit.circuit.library.standard_gates as gates
from qiskit.circuit import Gate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.utils import optionals
from .gate_sequence import GateSequence
def _check_candidate(candidate, existing_sequences, tol=1e-10):
    if optionals.HAS_SKLEARN:
        return _check_candidate_kdtree(candidate, existing_sequences, tol)
    warnings.warn("The SolovayKitaev algorithm relies on scikit-learn's KDTree for a fast search over the basis approximations. Without this, we fallback onto a greedy search with is significantly slower. We highly suggest to install scikit-learn to use this feature.", category=RuntimeWarning)
    return _check_candidate_greedy(candidate, existing_sequences, tol)