from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def update_A(self, A_new):
    """
        Update constraint matrix
        """
    if self.work.clear_update_time == 1:
        self.work.clear_update_time = 0
        self.work.info.update_time = 0.0
    self.work.timer = time.time()
    if self.work.settings.scaling:
        self.work.data.A = self.work.scaling.E.dot(A_new.dot(self.work.scaling.D))
    else:
        self.work.data.A = A_new
    self.work.linsys_solver = linsys_solver(self.work)
    self.work.info.update_time += time.time() - self.work.timer