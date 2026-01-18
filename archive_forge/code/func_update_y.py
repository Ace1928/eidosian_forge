from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def update_y(self):
    """
        Third ADMM step: update dual variable y
        """
    self.work.delta_y = self.work.rho_vec * (self.work.settings.alpha * self.work.xz_tilde[self.work.data.n:] + (1.0 - self.work.settings.alpha) * self.work.z_prev - self.work.z)
    self.work.y += self.work.delta_y