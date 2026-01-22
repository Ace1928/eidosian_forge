import abc
from typing import Tuple
import numpy as np
import cvxpy.interface.matrix_utilities
Formats the block for block_add.

        Args:
            matrix: The matrix the block will be added to.
            block: The matrix/scalar to be added.
            rows: The height of the block.
            cols: The width of the block.
        