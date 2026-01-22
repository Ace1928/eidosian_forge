from itertools import product, chain
from ..sage_helper import _within_sage
from ..math_basics import prod
from ..pari import pari
from .fundamental_polyhedron import *
def _build_hom_dict(self):
    gens, matrices = (self._gens, self._matrices)
    inv_gens = [g.upper() for g in gens]
    inv_mat = [SL2C_inverse(m) for m in matrices]
    self._hom_dict = dict(zip(gens + inv_gens, matrices + inv_mat))