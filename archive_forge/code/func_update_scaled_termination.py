from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def update_scaled_termination(self, scaled_termination_new):
    """
        Update scaled_termination parameter
        """
    if (scaled_termination_new is not True) & (scaled_termination_new is not False):
        raise ValueError('scaled_termination should be either True or False')
    self.work.settings.scaled_termination = scaled_termination_new