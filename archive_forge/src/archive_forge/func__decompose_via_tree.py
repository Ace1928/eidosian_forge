from typing import Sequence, Tuple
import numpy as np
from numpy.typing import NDArray
import attr
import cirq
from cirq._compat import cached_property
from cirq_ft import infra
def _decompose_via_tree(self, controls: NDArray[cirq.Qid], control_values: Sequence[int], ancillas: NDArray[cirq.Qid], target: cirq.Qid) -> cirq.ops.op_tree.OpTree:
    """Decomposes multi-controlled `And` in-terms of an `And` ladder of size #controls- 2."""
    if len(controls) == 2:
        yield And(control_values, adjoint=self.adjoint).on(*controls, target)
        return
    new_controls = np.concatenate([ancillas[0:1], controls[2:]])
    new_control_values = (1, *control_values[2:])
    and_op = And(control_values[:2], adjoint=self.adjoint).on(*controls[:2], ancillas[0])
    if self.adjoint:
        yield from self._decompose_via_tree(new_controls, new_control_values, ancillas[1:], target)
        yield and_op
    else:
        yield and_op
        yield from self._decompose_via_tree(new_controls, new_control_values, ancillas[1:], target)