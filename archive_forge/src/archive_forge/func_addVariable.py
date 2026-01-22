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
def addVariable(self, variable):
    """
        Adds a variable to the problem before a constraint is added

        @param variable: the variable to be added
        """
    if variable.hash not in self._variable_ids:
        self._variables.append(variable)
        self._variable_ids[variable.hash] = variable