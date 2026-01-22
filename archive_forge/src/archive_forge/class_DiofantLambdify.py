from __future__ import (absolute_import, division, print_function)
import os
import sys
import numpy as np
from .util import banded_jacobian, sparse_jacobian_csc, sparse_jacobian_csr
class DiofantLambdify(_Lambdify):

    def __init__(self, args, *exprs, **kwargs):
        kwargs['backend'] = 'diofant'
        super().__init__(args, *exprs, **kwargs)