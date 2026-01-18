from itertools import product, chain
from ..sage_helper import _within_sage
from ..math_basics import prod
from ..pari import pari
from .fundamental_polyhedron import *
def trace_field_generators(self):
    gens = self.generators()
    enough_elts = [''.join(sorted(s)) for s in powerset(gens) if len(s) > 0]
    return [self(w).trace() for w in enough_elts]