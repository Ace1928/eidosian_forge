from functools import lru_cache
from typing import List, Union
import numpy as np
from qiskit.circuit import Qubit
from qiskit.circuit.operation import Operation
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.quantum_info.operators import Operator
def _commutation_precheck(op1: Operation, qargs1: List, cargs1: List, op2: Operation, qargs2: List, cargs2: List, max_num_qubits):
    if getattr(op1, 'condition', None) is not None or getattr(op2, 'condition', None) is not None:
        return False
    if isinstance(op1, ControlFlowOp) or isinstance(op2, ControlFlowOp):
        return False
    intersection_q = set(qargs1).intersection(set(qargs2))
    intersection_c = set(cargs1).intersection(set(cargs2))
    if not (intersection_q or intersection_c):
        return True
    if len(qargs1) > max_num_qubits or len(qargs2) > max_num_qubits:
        return False
    if op1.name in _skipped_op_names or op2.name in _skipped_op_names:
        return False
    if getattr(op1, '_directive', False) or getattr(op2, '_directive', False):
        return False
    if getattr(op1, 'is_parameterized', False) and op1.is_parameterized() or (getattr(op2, 'is_parameterized', False) and op2.is_parameterized()):
        return False
    return None