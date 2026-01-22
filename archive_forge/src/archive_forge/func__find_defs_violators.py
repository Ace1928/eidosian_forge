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
def _find_defs_violators(blocks, cfg):
    """
    Returns
    -------
    res : Set[str]
        The SSA violators in a dictionary of variable names.
    """
    defs = defaultdict(list)
    uses = defaultdict(set)
    states = dict(defs=defs, uses=uses)
    _run_block_analysis(blocks, states, _GatherDefsHandler())
    _logger.debug('defs %s', pformat(defs))
    violators = {k for k, vs in defs.items() if len(vs) > 1}
    doms = cfg.dominators()
    for k, use_blocks in uses.items():
        if k not in violators:
            for label in use_blocks:
                dom = doms[label]
                def_labels = {label for _assign, label in defs[k]}
                if not def_labels.intersection(dom):
                    violators.add(k)
                    break
    _logger.debug('SSA violators %s', pformat(violators))
    return violators