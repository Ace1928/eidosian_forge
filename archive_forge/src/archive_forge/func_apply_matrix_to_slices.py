import dataclasses
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from cirq import protocols
from cirq.linalg import predicates
def apply_matrix_to_slices(target: np.ndarray, matrix: np.ndarray, slices: Sequence[_TSlice], *, out: Optional[np.ndarray]=None) -> np.ndarray:
    """Left-multiplies an NxN matrix onto N slices of a numpy array.

    One example is that the 4x4 matrix of a fractional SWAP gate can be expressed as

    $$
    \\begin{bmatrix}
      1 & & \\\\
        & X**t & \\\\
        & & 1 \\\\
    \\end{bmatrix}

    Where X is the 2x2 Pauli X gate and t is the power of the swap with t=1
    being a full swap. X**t is a power of the Pauli X gate's matrix.
    Applying the fractional swap is equivalent to applying a fractional X
    within the inner 2x2 subspace; the rest of the matrix is identity. This
    can be expressed using `apply_matrix_to_slices` as follows:

        def fractional_swap(target):
            assert target.shape == (4,)
            return apply_matrix_to_slices(
                target=target,
                matrix=cirq.unitary(cirq.X**t),
                slices=[1, 2]
            )

    Args:
        target: The input array with slices that need to be left-multiplied.
        matrix: The linear operation to apply to the subspace defined by the
            slices.
        slices: The parts of the tensor that correspond to the "vector entries"
            that the matrix should operate on. May be integers or complicated
            multi-dimensional slices into a tensor. The slices must refer to
            non-overlapping sections of the input all with the same shape.
        out: Where to write the output. If not specified, a new numpy array is
            created, with the same shape and dtype as the target, to store the
            output.

    Returns:
        The transformed array.

    Raises:
        ValueError: If `out` is `target` , or the matrix shaped does not match
            `slices`.
    """
    if out is target:
        raise ValueError("Can't write output over the input.")
    if matrix.shape != (len(slices), len(slices)):
        raise ValueError('matrix.shape != (len(slices), len(slices))')
    if out is None:
        out = np.copy(target)
    else:
        out[...] = target[...]
    for i, s_i in enumerate(slices):
        out[s_i] *= matrix[i, i]
        for j, s_j in enumerate(slices):
            if i != j:
                out[s_i] += target[s_j] * matrix[i, j]
    return out