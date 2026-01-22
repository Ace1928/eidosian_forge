import dataclasses
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from cirq import protocols
from cirq.linalg import predicates
def _build_from_slices(args: Sequence[_BuildFromSlicesArgs], source: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Populates `out` from the desired slices of `source`.

    This function is best described by example.

    For instance in 3*3*3 3D space, one could take a cube array, take all the horizontal slices,
    and add them up into the top slice leaving everything else zero. If the vertical axis was 1,
    and the top was index=2, then this would be written as follows:

        _build_from_slices(
            [
                _BuildFromSlicesArgs((_SliceConfig(axis=1, source_index=0, target_index=2),), 1),
                _BuildFromSlicesArgs((_SliceConfig(axis=1, source_index=1, target_index=2),), 1),
                _BuildFromSlicesArgs((_SliceConfig(axis=1, source_index=2, target_index=2),), 1),
            ],
            source,
            out,
        )

    When multiple slices are included in the _BuildFromSlicesArgs, this means to take the
    intersection of the source space and move it to the intersection of the target space. For
    example, the following takes the bottom-left edge and moves it to the top-right, leaving all
    other cells zero. Assume the lateral axis is 2 and right-most index thereof is 2:

        _build_from_slices(
            [
                _BuildFromSlicesArgs(
                    (
                        _SliceConfig(axis=1, source_index=0, target_index=2),  # top
                        _SliceConfig(axis=2, source_index=0, target_index=2),  # right
                    ),
                    scale=1,
                ),
            ],
            source,
            out,
        )

    This function is useful for optimizing multiplying a state by one or more one-hot matrices,
    as is common when working with Kraus components. It is more efficient than using an einsum.

    Args:
        args: The list of slice configurations to sum up into the output.
        source: The source tensor for the slice data.
        out: An output tensor that is the same shape as the source.

    Returns:
        The output tensor.
    """
    d = len(source.shape)
    out[...] = 0
    for arg in args:
        source_slice: List[Any] = [slice(None)] * d
        target_slice: List[Any] = [slice(None)] * d
        for sleis in arg.slices:
            source_slice[sleis.axis] = sleis.source_index
            target_slice[sleis.axis] = sleis.target_index
        out[tuple(target_slice)] += arg.scale * source[tuple(source_slice)]
    return out