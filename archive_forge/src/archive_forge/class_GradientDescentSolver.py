from __future__ import (absolute_import, division, print_function)
import math
import numpy as np
class GradientDescentSolver(SolverBase):
    """ Example of a custom solver

    Parameters
    ----------
    instance: NeqSys instance (passed by NeqSys.solve)

    Attributes
    ----------
    damping: callback
        with signature f(iter_idx, iter_max) -> float
        default: lambda idx, mx: exp(-idx/mx)/2

    Notes
    -----
    Note that this is an inefficient solver for demonstration purposes.
    """

    def damping(self, iter_idx, mx_iter):
        return 1.0

    def step(self, x, iter_idx, maxiter):
        return self.damping(iter_idx, maxiter) * self._gd_step(x)