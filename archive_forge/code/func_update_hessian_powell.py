import time
import numpy as np
from numpy.linalg import eigh
from ase.optimize.optimize import Optimizer
def update_hessian_powell(self, pos, G):
    n = len(self.hessian)
    dgrad = G - self.oldG
    dpos = pos - self.oldpos
    absdpos = np.dot(dpos, dpos)
    if absdpos < self.eps:
        return
    dotg = np.dot(dgrad, dpos)
    tvec = dgrad - np.dot(dpos, self.hessian)
    tvecdpos = np.dot(tvec, dpos)
    ddot = tvecdpos / absdpos
    dott = np.dot(dpos, tvec)
    if abs(dott) > self.eps and abs(dotg) > self.eps:
        for i in range(n):
            for j in range(n):
                h = tvec[i] * dpos[j] + dpos[i] * tvec[j] - ddot * dpos[i] * dpos[j]
                h *= 1.0 / absdpos
                self.hessian[i][j] += h