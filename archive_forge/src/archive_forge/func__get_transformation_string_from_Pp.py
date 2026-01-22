from __future__ import annotations
import re
from fractions import Fraction
from typing import TYPE_CHECKING
import numpy as np
from sympy import Matrix
from sympy.parsing.sympy_parser import parse_expr
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import MagSymmOp, SymmOp
from pymatgen.util.string import transformation_to_string
@staticmethod
def _get_transformation_string_from_Pp(P: list[list[float]] | np.ndarray, p: list[float]) -> str:
    P = np.array(P).transpose()
    P_string = transformation_to_string(P, components=('a', 'b', 'c'))
    p_string = transformation_to_string(np.zeros((3, 3)), p)
    return P_string + ';' + p_string