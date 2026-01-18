from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def warm_start_y(self, y):
    """
        Warm start dual variable
        """
    self.work.settings.warm_start = True
    self.work.y = y
    self.work.y = self.work.scaling.Einv.dot(self.work.y)
    self.work.x = np.zeros(self.work.data.n)
    self.work.z = np.zeros(self.work.data.m)