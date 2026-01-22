import dataclasses
import typing
from typing import Callable, Optional
import cupy
from cupyx.distributed.array import _array
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _modes
Create an ``index_map`` for a 2D matrix with a specified blocking.

    Args:
        i_partitions (list of ints): boundaries of blocks on the `i` axis
        j_partitions (list of ints): boundaries of blocks on the `j` axis
        devices (2D list of sets of ints): devices owning each block

    Returns:
        dict from int to array indices: index_map
            Indices for the chunks that devices with designated IDs are going
            to own.

    Example:
        >>> index_map = make_2d_index_map(
        ...     [0, 2, 4], [0, 3, 5],
        ...     [[{0}, {1}],
        ...      [{2}, {0, 1}]])
        >>> pprint(index_map)
        {0: [(slice(0, 2, None), slice(0, 3, None)),
             (slice(2, 4, None), slice(3, 5, None))],
         1: [(slice(0, 2, None), slice(3, 5, None)),
             (slice(2, 4, None), slice(3, 5, None))],
         2: [(slice(2, 4, None), slice(0, 3, None))]}
    