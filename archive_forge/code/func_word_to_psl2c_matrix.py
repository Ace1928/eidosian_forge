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
def word_to_psl2c_matrix(mcomplex: Mcomplex, word: str):
    """
    Given a triangulation with a R13 geometric structure (that is
    the structure attached by calling add_r13_geometry) and a word
    in the simplified fundamental group (given as string), returns
    the corresponding PSL(2,C)-matrix.
    """
    return word_list_to_psl2c_matrix(mcomplex, word_as_list(word, mcomplex.num_generators))