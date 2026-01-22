import functools
import logging
import sys
import time
import traceback
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.modeling import NOTSET as _NotSpecified
def _move_grandchildren_to_root(root, child):
    """A helper function to assist with flattening of HierarchicalTimer
    objects

    Parameters
    ----------
    root: HierarchicalTimer or _HierarchicalHelper
        The root node. Children of `child` will become children of
        this node

    child: _HierarchicalHelper
        The child node that will be turned into a leaf by moving
        its children to the root

    """
    for gchild_key, gchild_timer in child.timers.items():
        if gchild_key in root.timers:
            gchild_total_time = gchild_timer.total_time
            gchild_n_calls = gchild_timer.n_calls
            root.timers[gchild_key].total_time += gchild_total_time
            root.timers[gchild_key].n_calls += gchild_n_calls
        else:
            root.timers[gchild_key] = gchild_timer
        child.total_time -= gchild_timer.total_time
    child.timers.clear()