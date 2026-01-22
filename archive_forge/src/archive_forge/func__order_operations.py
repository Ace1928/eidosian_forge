from functools import lru_cache
from typing import List, Union
import numpy as np
from qiskit.circuit import Qubit
from qiskit.circuit.operation import Operation
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.quantum_info.operators import Operator
def _order_operations(op1: Operation, qargs1: List, cargs1: List, op2: Operation, qargs2: List, cargs2: List):
    """Orders two operations in a canonical way that is persistent over
    @different python versions and executions
    Args:
        op1: first operation.
        qargs1: first operation's qubits.
        cargs1: first operation's clbits.
        op2: second operation.
        qargs2: second operation's qubits.
        cargs2: second operation's clbits.
    Return:
        The input operations in a persistent, canonical order.
    """
    op1_tuple = (op1, qargs1, cargs1)
    op2_tuple = (op2, qargs2, cargs2)
    least_qubits_op, most_qubits_op = (op1_tuple, op2_tuple) if op1.num_qubits < op2.num_qubits else (op2_tuple, op1_tuple)
    if op1.num_qubits != op2.num_qubits:
        return (least_qubits_op, most_qubits_op)
    else:
        return (op1_tuple, op2_tuple) if _persistent_id(op1.name) < _persistent_id(op2.name) else (op2_tuple, op1_tuple)