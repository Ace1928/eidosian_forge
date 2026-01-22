from sympy.core.numbers import igcd, mod_inverse
from sympy.core.power import integer_nthroot
from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power
from sympy.ntheory import isprime
from math import log, sqrt
import random
def _build_matrix(smooth_relations):
    """Build a 2D matrix from smooth relations.

    Parameters
    ==========

    smooth_relations : Stores smooth relations
    """
    matrix = []
    for s_relation in smooth_relations:
        matrix.append(s_relation[2])
    return matrix