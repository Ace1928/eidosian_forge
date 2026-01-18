import time
import numpy as np
from numpy.linalg import eigh
from ase.optimize.optimize import Optimizer
def update_hessian(self, pos, G):
    import copy
    if hasattr(self, 'oldG'):
        if self.hessianupdate == 'BFGS':
            self.update_hessian_bfgs(pos, G)
        elif self.hessianupdate == 'Powell':
            self.update_hessian_powell(pos, G)
        else:
            self.update_hessian_bofill(pos, G)
    elif not hasattr(self, 'hessian'):
        self.set_default_hessian()
    self.oldpos = copy.copy(pos)
    self.oldG = copy.copy(G)
    if self.verbosity:
        print('hessian ', self.hessian)