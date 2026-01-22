from __future__ import absolute_import, print_function
import sys
import inspect
from . import TypeSlots
from . import Builtin
from . import Nodes
from . import ExprNodes
from . import Errors
from . import DebugFlags
from . import Future
import cython
@cython.final
def _flatten_list(self, orig_list):
    newlist = []
    for x in orig_list:
        if x is not None:
            if type(x) is list:
                newlist.extend(x)
            else:
                newlist.append(x)
    return newlist