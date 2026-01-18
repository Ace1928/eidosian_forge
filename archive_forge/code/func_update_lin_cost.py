from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def update_lin_cost(self, q_new):
    """
        Update linear cost without requiring factorization
        """
    if self.work.clear_update_time == 1:
        self.work.clear_update_time = 0
        self.work.info.update_time = 0.0
    self.work.timer = time.time()
    self.work.data.q = np.copy(q_new)
    if self.work.settings.scaling:
        self.work.data.q = self.work.scaling.c * self.work.scaling.D.dot(self.work.data.q)
    self.reset_info(self.work.info)
    self.work.info.update_time += time.time() - self.work.timer