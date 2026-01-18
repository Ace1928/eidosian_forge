from math import prod
from sympy import QQ, ZZ
from sympy.abc import x, theta
from sympy.ntheory import factorint
from sympy.ntheory.residue_ntheory import n_order
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.matrices import DomainMatrix
from sympy.polys.numberfields.basis import round_two
from sympy.polys.numberfields.exceptions import StructureError
from sympy.polys.numberfields.modules import PowerBasis, to_col
from sympy.polys.numberfields.primes import (
from sympy.testing.pytest import raises
Useful for inspection, when running test manually.