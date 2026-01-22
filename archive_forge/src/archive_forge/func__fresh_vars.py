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
def _fresh_vars(blocks, varname):
    """Rewrite to put fresh variable names
    """
    states = _make_states(blocks)
    states['varname'] = varname
    states['defmap'] = defmap = defaultdict(list)
    newblocks = _run_block_rewrite(blocks, states, _FreshVarHandler())
    return (newblocks, defmap)