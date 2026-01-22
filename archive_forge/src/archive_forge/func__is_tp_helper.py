from __future__ import annotations
import copy
import sys
from abc import abstractmethod
from numbers import Number, Integral
import numpy as np
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.linear_op import LinearOp
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.predicates import is_positive_semidefinite_matrix
from qiskit.quantum_info.operators.channel.transformations import _transform_rep
from qiskit.quantum_info.operators.channel.transformations import _to_choi
from qiskit.quantum_info.operators.channel.transformations import _to_kraus
from qiskit.quantum_info.operators.channel.transformations import _to_operator
from qiskit.quantum_info.operators.scalar_op import ScalarOp
def _is_tp_helper(self, choi, atol, rtol):
    """Test if Choi-matrix is trace-preserving (TP)"""
    if atol is None:
        atol = self.atol
    if rtol is None:
        rtol = self.rtol
    d_in, d_out = self.dim
    mat = np.trace(np.reshape(choi, (d_in, d_out, d_in, d_out)), axis1=1, axis2=3)
    tp_cond = np.linalg.eigvalsh(mat - np.eye(len(mat)))
    zero = np.isclose(tp_cond, 0, atol=atol, rtol=rtol)
    return np.all(zero)