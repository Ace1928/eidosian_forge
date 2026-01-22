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
def addterm(self, key, value):
    y = self.get(key, 0)
    if y:
        y += value
        self[key] = y
    else:
        self[key] = value