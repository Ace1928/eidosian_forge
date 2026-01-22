import logging
import operator
import warnings
from functools import reduce
from copy import copy
from pprint import pformat
from collections import defaultdict
from numba import config
from numba.core import ir, ir_utils, errors
from numba.core.analysis import compute_cfg_from_blocks
def _fix_var(self, states, stmt, used_vars):
    """Fix all variable uses in ``used_vars``.
        """
    varnames = [k.name for k in used_vars]
    phivar = states['varname']
    if phivar in varnames:
        return self._find_def(states, stmt)