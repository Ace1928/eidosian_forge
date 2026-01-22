from itertools import product, chain
from ..sage_helper import _within_sage
from ..math_basics import prod
from ..pari import pari
from .fundamental_polyhedron import *
def SL2C_inverse(A):
    return matrix([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]])