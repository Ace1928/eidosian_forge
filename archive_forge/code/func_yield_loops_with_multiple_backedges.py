from collections import namedtuple, defaultdict
import logging
import operator
from numba.core.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.core import errors, ir, ir_utils
from numba.core.analysis import compute_use_defs, compute_cfg_from_blocks
from numba.core.utils import PYVERSION
def yield_loops_with_multiple_backedges():
    for lp in cfg.loops().values():
        if has_multiple_backedges(lp):
            yield lp