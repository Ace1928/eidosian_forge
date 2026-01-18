from .line import R13LineWithMatrix
from ..verify.shapes import compute_hyperbolic_shapes # type: ignore
from ..snap.fundamental_polyhedron import FundamentalPolyhedronEngine # type: ignore
from ..snap.kernel_structures import TransferKernelStructuresEngine # type: ignore
from ..snap.t3mlite import simplex, Mcomplex, Tetrahedron, Vertex # type: ignore
from ..SnapPy import word_as_list # type: ignore
from ..hyperboloid import (o13_inverse,  # type: ignore
from ..upper_halfspace import sl2c_inverse, psl2c_to_o13 # type: ignore
from ..upper_halfspace.ideal_point import ideal_point_to_r13 # type: ignore
from ..matrix import vector, matrix, mat_solve # type: ignore
from ..math_basics import prod, xgcd # type: ignore
from collections import deque
from typing import Tuple, Sequence, Optional, Any
def word_list_to_psl2c_matrix(mcomplex: Mcomplex, word_list: Sequence[int]):
    """
    Like word_to_psl2c_matrix, but taking the word as a sequence of
    non-zero integers with positive integers corresponding to generators and
    negative integers corresponding to their inverses.
    """
    return prod([mcomplex.GeneratorMatrices[g] for g in word_list])