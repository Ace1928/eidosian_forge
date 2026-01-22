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
def _findValue(self, attrib):
    """
        safe way to get the value of a variable that may not exist
        """
    var = getattr(self, attrib, 0)
    if var:
        if value(var) is not None:
            return value(var)
        else:
            return 0.0
    else:
        return 0.0