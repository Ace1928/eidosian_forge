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
def transform_lattice(self, lattice: Lattice) -> Lattice:
    """Transforms a lattice."""
    return Lattice(np.matmul(lattice.matrix, self.P))