import warnings
from typing import Any, List, Optional, Sequence, Union
import numpy as np
from numpy.random.mtrand import RandomState
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.pyqvm import AbstractQuantumSimulator
from pyquil.quilbase import Gate
from pyquil.simulation.matrices import P0, P1, KRAUS_OPS, QUANTUM_GATES
from pyquil.simulation.tools import lifted_gate_matrix, lifted_gate, all_bitstrings
def _is_valid_quantum_state(state_matrix: np.ndarray, rtol: float=1e-05, atol: float=1e-08) -> bool:
    """
    Checks if a quantum state is valid, i.e. the matrix is Hermitian; trace one, and that the
    eigenvalues are non-negative.

    :param state_matrix: a D by D np.ndarray representing a quantum state
    :param rtol: The relative tolerance parameter in np.allclose and np.isclose
    :param atol: The absolute tolerance parameter in np.allclose and np.isclose
    :return: bool
    """
    hermitian = np.allclose(state_matrix, np.conjugate(state_matrix.transpose()), rtol, atol)
    if not hermitian:
        raise ValueError('The state matrix is not Hermitian.')
    trace_one = np.isclose(np.trace(state_matrix), 1, rtol, atol)
    if not trace_one:
        raise ValueError('The state matrix is not trace one.')
    evals = np.linalg.eigvals(state_matrix)
    non_neg_eigs = all([False if val < -atol else True for val in evals])
    if not non_neg_eigs:
        raise ValueError('The state matrix has negative Eigenvalues of order -' + str(atol) + '.')
    return hermitian and trace_one and non_neg_eigs