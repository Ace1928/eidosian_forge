from sympy.core.add import Add
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory.generate import prime
from sympy.polys.domains.integerring import ZZ
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import permute_signs
from sympy.testing.pytest import raises
from sympy.polys.specialpolys import (
from sympy.abc import x, y, z
Tests for functions for generating interesting polynomials. 