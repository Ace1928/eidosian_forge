from __future__ import annotations
from pyomo.common.dependencies import mpi4py
from .mpi_block_vector import MPIBlockVector
from .block_vector import BlockVector
from .block_matrix import BlockMatrix, NotFullyDefinedBlockMatrixError
from .block_matrix import assert_block_structure as block_matrix_assert_block_structure
from .base_block import BaseBlockMatrix
import numpy as np
from scipy.sparse import coo_matrix
import operator

            Consider a non-empty block m_{i, j} from the mpi block matrix with rank owner r_m and the
            corresponding block v_{j} from the mpi block vector with rank owner r_v. There are 4 cases:
              1. r_m = r_v
                 In this case, all is good.
              2. r_v = -1
                 In this case, all is good.
              3. r_m = -1 and r_v = 0
                 All is good
              4. If none of the above cases hold, then v_{j} must be broadcast
            