from itertools import product, chain
from ..sage_helper import _within_sage
from ..math_basics import prod
from ..pari import pari
from .fundamental_polyhedron import *
def twist(g, gen_image):
    return gen_image if phi(g)[0] % 2 == 0 else -gen_image