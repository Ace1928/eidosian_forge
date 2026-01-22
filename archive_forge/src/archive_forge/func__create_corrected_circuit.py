from typing import Iterator, List, Optional
import itertools
import math
import numpy as np
import cirq
from cirq_google import ops
def _create_corrected_circuit(target_unitary: np.ndarray, program: cirq.Circuit, q0: cirq.Qid, q1: cirq.Qid) -> cirq.OP_TREE:
    """Adds pre/post single qubit rotations to `program` to make it equivalent to `target_unitary`.

    Adds single qubit correction terms to the given circuit on 2 qubit s.t. it implements
    `target_unitary`. This assumes that `program` implements a 2q unitary effect which has same
    interaction coefficients as `target_unitary` in it's KAK decomposition and differs only in
    local unitary rotations.

    Args:
        target_unitary: The unitary that should be implemented by the transformed `program`.
        program: `cirq.Circuit` to be transformed.
        q0: First qubit to operate on.
        q1: Second qubit to operate on.

    Yields:
        Operations in `program` with pre- and post- rotations added s.t. the resulting
        `cirq.OP_TREE`
        implements `target_unitary`.
    """
    b_0, b_1, a_0, a_1 = _find_local_equivalents(target_unitary, program.unitary(qubit_order=cirq.QubitOrder.explicit([q0, q1]), dtype=np.complex128))
    yield from _phased_x_z_ops(b_0, q0)
    yield from _phased_x_z_ops(b_1, q1)
    yield program
    yield from _phased_x_z_ops(a_0, q0)
    yield from _phased_x_z_ops(a_1, q1)