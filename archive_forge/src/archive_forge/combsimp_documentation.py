from sympy.core import Mul
from sympy.core.function import count_ops
from sympy.core.traversal import preorder_traversal, bottom_up
from sympy.functions.combinatorial.factorials import binomial, factorial
from sympy.functions import gamma
from sympy.simplify.gammasimp import gammasimp, _gammasimp
from sympy.utilities.timeutils import timethis

    Helper function for combsimp.

    Rewrites expression in terms of factorials and binomials
    