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
class _GatherDefsHandler(_BaseHandler):
    """Find all defs and uses of variable in each block

    ``states["label"]`` is a int; label of the current block
    ``states["defs"]`` is a Mapping[str, List[Tuple[ir.Assign, int]]]:
        - a mapping of the name of the assignee variable to the assignment
          IR node and the block label.
    ``states["uses"]`` is a Mapping[Set[int]]
    """

    def on_assign(self, states, assign):
        states['defs'][assign.target.name].append((assign, states['label']))
        for var in assign.list_vars():
            k = var.name
            if k != assign.target.name:
                states['uses'][k].add(states['label'])

    def on_other(self, states, stmt):
        for var in stmt.list_vars():
            k = var.name
            states['uses'][k].add(states['label'])