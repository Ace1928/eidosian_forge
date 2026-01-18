from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def update_polish(self, polish_new):
    """
        Update polish parameter
        """
    if (polish_new is not True) & (polish_new is not False):
        raise ValueError('polish should be either True or False')
    self.work.settings.polish = polish_new
    self.work.info.polish_time = 0.0