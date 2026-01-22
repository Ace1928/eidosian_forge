from __future__ import annotations
from typing import Union, Optional
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.synthesis.evolution import EvolutionSynthesis, LieTrotter
from qiskit.quantum_info import Pauli, SparsePauliOp
def _to_sparse_pauli_op(operator):
    """Cast the operator to a SparsePauliOp."""
    if isinstance(operator, Pauli):
        sparse_pauli = SparsePauliOp(operator)
    elif isinstance(operator, SparsePauliOp):
        sparse_pauli = operator
    else:
        raise ValueError(f'Unsupported operator type for evolution: {type(operator)}.')
    if any(np.iscomplex(sparse_pauli.coeffs)):
        raise ValueError('Operator contains complex coefficients, which are not supported.')
    if any((isinstance(coeff, ParameterExpression) for coeff in sparse_pauli.coeffs)):
        raise ValueError('Operator contains ParameterExpression, which are not supported.')
    return sparse_pauli