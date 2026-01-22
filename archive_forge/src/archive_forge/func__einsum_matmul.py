from __future__ import annotations
import copy
import re
from numbers import Number
from typing import TYPE_CHECKING
import numpy as np
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.library.standard_gates import HGate, IGate, SGate, TGate, XGate, YGate, ZGate
from qiskit.circuit.operation import Operation
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.linear_op import LinearOp
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.predicates import is_unitary_matrix, matrix_equal
@classmethod
def _einsum_matmul(cls, tensor, mat, indices, shift=0, right_mul=False):
    """Perform a contraction using Numpy.einsum

        Args:
            tensor (np.array): a vector or matrix reshaped to a rank-N tensor.
            mat (np.array): a matrix reshaped to a rank-2M tensor.
            indices (list): tensor indices to contract with mat.
            shift (int): shift for indices of tensor to contract [Default: 0].
            right_mul (bool): if True right multiply tensor by mat
                              (else left multiply) [Default: False].

        Returns:
            Numpy.ndarray: the matrix multiplied rank-N tensor.

        Raises:
            QiskitError: if mat is not an even rank tensor.
        """
    rank = tensor.ndim
    rank_mat = mat.ndim
    if rank_mat % 2 != 0:
        raise QiskitError('Contracted matrix must have an even number of indices.')
    indices_tensor = list(range(rank))
    for j, index in enumerate(indices):
        indices_tensor[index + shift] = rank + j
    mat_contract = list(reversed(range(rank, rank + len(indices))))
    mat_free = [index + shift for index in reversed(indices)]
    if right_mul:
        indices_mat = mat_contract + mat_free
    else:
        indices_mat = mat_free + mat_contract
    return np.einsum(tensor, indices_tensor, mat, indices_mat)