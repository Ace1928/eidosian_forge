from contextlib import contextmanager
from threading import local
from sympy.core.function import expand_mul
class DotProdSimpState(local):

    def __init__(self):
        self.state = None