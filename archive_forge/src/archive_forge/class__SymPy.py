from __future__ import (absolute_import, division, print_function)
import os
import sys
import numpy as np
from .util import banded_jacobian, sparse_jacobian_csc, sparse_jacobian_csr
class _SymPy(_Base):

    def __init__(self):
        self.__sym_backend__ = __import__('sympy')
        from ._sympy_Lambdify import _Lambdify
        self.Lambdify = _Lambdify

    def real_symarray(self, prefix, shape):
        return self.symarray(prefix, shape, real=True)
    DenseMatrix = _DenseMatrix