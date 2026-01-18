import numpy as np
import scipy.sparse as sp
from cvxpy.interface.numpy_interface.ndarray_interface import NDArrayInterface
Add the block to a slice of the matrix.

        Args:
            matrix: The matrix the block will be added to.
            block: The matrix/scalar to be added.
            vert_offset: The starting row for the matrix slice.
            horiz_offset: The starting column for the matrix slice.
            rows: The height of the block.
            cols: The width of the block.
            vert_step: The row step size for the matrix slice.
            horiz_step: The column step size for the matrix slice.
        