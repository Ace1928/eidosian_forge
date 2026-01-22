from functools import lru_cache
from typing import List, Union
import numpy as np
from qiskit.circuit import Qubit
from qiskit.circuit.operation import Operation
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.quantum_info.operators import Operator
def _get_relative_placement(first_qargs: List[Qubit], second_qargs: List[Qubit]) -> tuple:
    """Determines the relative qubit placement of two gates. Note: this is NOT symmetric.

    Args:
        first_qargs (DAGOpNode): first gate
        second_qargs (DAGOpNode): second gate

    Return:
        A tuple that describes the relative qubit placement. The relative placement is defined by the
        gate qubit arrangements as q2^{-1}[q1[i]] where q1[i] is the ith qubit of the first gate and
        q2^{-1}[q] returns the qubit index of qubit q in the second gate (possibly 'None'). E.g.
        _get_relative_placement(CX(0, 1), CX(1, 2)) would return (None, 0) as there is no overlap on
        the first qubit of the first gate but there is an overlap on the second qubit of the first gate,
        i.e. qubit 0 of the second gate. Likewise, _get_relative_placement(CX(1, 2), CX(0, 1)) would
        return (1, None)
    """
    qubits_g2 = {q_g1: i_g1 for i_g1, q_g1 in enumerate(second_qargs)}
    return tuple((qubits_g2.get(q_g0, None) for q_g0 in first_qargs))