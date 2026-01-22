from __future__ import annotations
from string import ascii_uppercase, ascii_lowercase
import numpy as np
import qiskit.circuit.library.standard_gates as gates
from qiskit.exceptions import QiskitError
def _einsum_matmul_index_helper(gate_indices: list[int], number_of_qubits: int) -> tuple[str, str, str, str]:
    """Return the index string for Numpy.einsum matrix multiplication.

    The returned indices are to perform a matrix multiplication A.v where
    the matrix A is an M-qubit matrix, matrix v is an N-qubit vector, and
    M <= N, and identity matrices are implied on the subsystems where A has no
    support on v.

    Args:
        gate_indices (list[int]): the indices of the right matrix subsystems
                                   to contract with the left matrix.
        number_of_qubits (int): the total number of qubits for the right matrix.

    Returns:
        tuple: (mat_left, mat_right, tens_in, tens_out) of index strings for
        that may be combined into a Numpy.einsum function string.

    Raises:
        QiskitError: if the total number of qubits plus the number of
        contracted indices is greater than 26.
    """
    if len(gate_indices) + number_of_qubits > 26:
        raise QiskitError('Total number of free indexes limited to 26')
    tens_in = ascii_lowercase[:number_of_qubits]
    tens_out = list(tens_in)
    mat_left = ''
    mat_right = ''
    for pos, idx in enumerate(reversed(gate_indices)):
        mat_left += ascii_lowercase[-1 - pos]
        mat_right += tens_in[-1 - idx]
        tens_out[-1 - idx] = ascii_lowercase[-1 - pos]
    tens_out = ''.join(tens_out)
    return (mat_left, mat_right, tens_in, tens_out)