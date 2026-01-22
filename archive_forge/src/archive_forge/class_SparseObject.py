from itertools import count
from functools import reduce
from .tracer import trace, primitive, toposort, Node, Box, isbox, getval
from .util import func, subval
import warnings
class SparseObject(object):
    __slots__ = ['vs', 'mut_add']

    def __init__(self, vs, mut_add):
        self.vs = vs
        self.mut_add = mut_add