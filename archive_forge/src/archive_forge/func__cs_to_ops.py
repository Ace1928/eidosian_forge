from typing import Union, Tuple, Sequence, List, Optional
import numpy as np
import cirq
from cirq import ops
from cirq import transformers as opt
def _cs_to_ops(q0: ops.Qid, q1: ops.Qid, q2: ops.Qid, theta: np.ndarray) -> List[ops.Operation]:
    """Converts theta angles based Cosine Sine matrix to operations.

    Using the optimization as per Appendix A.1, it uses CZ gates instead of
    CNOT gates and returns a circuit that skips the terminal CZ gate.

    Args:
        q0: first qubit
        q1: second qubit
        q2: third qubit
        theta: theta returned from the Cosine Sine decomposition

    Returns:
         the operations
    """
    angles = _multiplexed_angles(theta * 2)
    rys = [cirq.ry(angle).on(q0) for angle in angles]
    ops = [rys[0], cirq.CZ(q1, q0), rys[1], cirq.CZ(q2, q0), rys[2], cirq.CZ(q1, q0), rys[3], cirq.CZ(q2, q0)]
    return _optimize_multiplexed_angles_circuit(ops)