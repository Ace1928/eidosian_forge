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
def _run_block_analysis(blocks, states, handler):
    for label, blk in blocks.items():
        _logger.debug('==== SSA block analysis pass on %s', label)
        states['label'] = label
        for _ in _run_ssa_block_pass(states, blk, handler):
            pass