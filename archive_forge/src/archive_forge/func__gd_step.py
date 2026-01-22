from __future__ import (absolute_import, division, print_function)
import math
import numpy as np
def _gd_step(self, x):
    self.cur_j = self.j(x)
    return -self.cur_j.dot(self.history_f[-1])