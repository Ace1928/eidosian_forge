from __future__ import print_function
from builtins import range
from builtins import object
import numpy as np
import scipy as sp
import scipy.sparse as spspa
import scipy.sparse.linalg as spla
import numpy.linalg as la
import time   # Time execution
def update_check_termination(self, check_termination_new):
    """
        Update check_termination parameter
        """
    if check_termination_new <= 0:
        raise ValueError('check_termination should be greater than 0')
    self.work.settings.check_termination = check_termination_new