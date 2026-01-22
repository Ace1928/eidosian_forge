from typing import Sequence, Tuple
import numpy as np
from numpy.typing import NDArray
import attr
import cirq
from cirq._compat import cached_property
from cirq_ft import infra
def _decompose_single_and(self, cv1: int, cv2: int, c1: cirq.Qid, c2: cirq.Qid, target: cirq.Qid) -> cirq.ops.op_tree.OpTree:
    """Decomposes a single `And` gate on 2 controls and 1 target in terms of Clifford+T gates.

        * And(cv).on(c1, c2, target) uses 4 T-gates and assumes target is in |0> state.
        * And(cv, adjoint=True).on(c1, c2, target) uses measurement based un-computation
            (0 T-gates) and will always leave the target in |0> state.
        """
    pre_post_ops = [cirq.X(q) for q, v in zip([c1, c2], [cv1, cv2]) if v == 0]
    yield pre_post_ops
    if self.adjoint:
        yield cirq.H(target)
        yield cirq.measure(target, key=f'{target}')
        yield cirq.CZ(c1, c2).with_classical_controls(f'{target}')
        yield cirq.reset(target)
    else:
        yield [cirq.H(target), cirq.T(target)]
        yield [cirq.CNOT(c1, target), cirq.CNOT(c2, target)]
        yield [cirq.CNOT(target, c1), cirq.CNOT(target, c2)]
        yield [cirq.T(c1) ** (-1), cirq.T(c2) ** (-1), cirq.T(target)]
        yield [cirq.CNOT(target, c1), cirq.CNOT(target, c2)]
        yield [cirq.H(target), cirq.S(target)]
    yield pre_post_ops