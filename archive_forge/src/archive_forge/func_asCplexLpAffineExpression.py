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
def asCplexLpAffineExpression(self, name, constant=1):
    """
        returns a string that represents the Affine Expression in lp format
        """
    result, line = self.asCplexVariablesOnly(name)
    if not self:
        term = f' {self.constant}'
    else:
        term = ''
        if constant:
            if self.constant < 0:
                term = ' - %s' % -self.constant
            elif self.constant > 0:
                term = f' + {self.constant}'
    if self._count_characters(line) + len(term) > const.LpCplexLPLineSize:
        result += [''.join(line)]
        line = [term]
    else:
        line += [term]
    result += [''.join(line)]
    result = '%s\n' % '\n'.join(result)
    return result