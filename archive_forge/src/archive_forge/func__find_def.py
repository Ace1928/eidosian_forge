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
def _find_def(self, states, stmt):
    """Find definition of ``stmt`` for the statement ``stmt``
        """
    _logger.debug('find_def var=%r stmt=%s', states['varname'], stmt)
    selected_def = None
    label = states['label']
    local_defs = states['defmap'][label]
    local_phis = states['phimap'][label]
    block = states['block']
    cur_pos = self._stmt_index(stmt, block)
    for defstmt in reversed(local_defs):
        def_pos = self._stmt_index(defstmt, block, stop=cur_pos)
        if def_pos < cur_pos:
            selected_def = defstmt
            break
        elif defstmt in local_phis:
            selected_def = local_phis[-1]
            break
    if selected_def is None:
        selected_def = self._find_def_from_top(states, label, loc=stmt.loc)
    return selected_def