import time
import numpy as np
from numpy.linalg import eigh
from ase.optimize.optimize import Optimizer
def update_hessian_bfgs(self, pos, G):
    n = len(self.hessian)
    dgrad = G - self.oldG
    dpos = pos - self.oldpos
    dotg = np.dot(dgrad, dpos)
    tvec = np.dot(dpos, self.hessian)
    dott = np.dot(dpos, tvec)
    if abs(dott) > self.eps and abs(dotg) > self.eps:
        for i in range(n):
            for j in range(n):
                h = dgrad[i] * dgrad[j] / dotg - tvec[i] * tvec[j] / dott
                self.hessian[i][j] += h