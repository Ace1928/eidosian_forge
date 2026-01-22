from typing import List, TYPE_CHECKING
import numpy as np
from cirq import ops, qis, circuits
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
def _1q_matrices_to_ops(g0, g1, q0, q1, include_identity=False):
    ret = []
    for g, q in zip(map(single_qubit_decompositions.single_qubit_matrix_to_phxz, [g0, g1]), [q0, q1]):
        if g is not None:
            ret.append(g.on(q))
        elif include_identity:
            ret.append(ops.I.on(q))
    return ret