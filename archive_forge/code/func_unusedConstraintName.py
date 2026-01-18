from collections import Counter
import sys
import warnings
from time import time
from .apis import LpSolverDefault, PULP_CBC_CMD
from .apis.core import clock
from .utilities import value
from . import constants as const
from . import mps_lp as mpslp
import logging
import re
def unusedConstraintName(self):
    self.lastUnused += 1
    while 1:
        s = '_C%d' % self.lastUnused
        if s not in self.constraints:
            break
        self.lastUnused += 1
    return s