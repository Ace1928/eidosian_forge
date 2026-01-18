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
@property
def transformation_string(self) -> str:
    """Transformation string."""
    return self._get_transformation_string_from_Pp(self.P, self.p)