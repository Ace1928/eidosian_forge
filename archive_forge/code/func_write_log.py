import time
import numpy as np
from numpy.linalg import eigh
from ase.optimize.optimize import Optimizer
def write_log(self, text):
    if self.logfile is not None:
        self.logfile.write(text + '\n')
        self.logfile.flush()