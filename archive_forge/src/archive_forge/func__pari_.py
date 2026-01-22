from ..sage_helper import _within_sage
from snappy.number import SnapPyNumbers, Number
from itertools import chain
from ..pari import pari, PariError
from .fundamental_polyhedron import Infinity
def _pari_(self):
    raise PariError