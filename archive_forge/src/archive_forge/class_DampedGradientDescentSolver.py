from __future__ import (absolute_import, division, print_function)
import math
import numpy as np
class DampedGradientDescentSolver(GradientDescentSolver):

    def __init__(self, base_damp=0.5, exp_damp=0.5):
        self.base_damp = base_damp
        self.exp_damp = exp_damp

    def damping(self, iter_idx, mx_iter):
        return self.base_damp * math.exp(-iter_idx / mx_iter * self.exp_damp)