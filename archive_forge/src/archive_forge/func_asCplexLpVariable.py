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
def asCplexLpVariable(self):
    if self.isFree():
        return self.name + ' free'
    if self.isConstant():
        return self.name + f' = {self.lowBound:.12g}'
    if self.lowBound == None:
        s = '-inf <= '
    elif self.lowBound == 0 and self.cat == const.LpContinuous:
        s = ''
    else:
        s = f'{self.lowBound:.12g} <= '
    s += self.name
    if self.upBound is not None:
        s += f' <= {self.upBound:.12g}'
    return s