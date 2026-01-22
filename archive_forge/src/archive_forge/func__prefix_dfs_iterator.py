import copy
import logging
import sys
import weakref
import textwrap
from collections import defaultdict
from contextlib import contextmanager
from inspect import isclass, currentframe
from io import StringIO
from itertools import filterfalse, chain
from operator import itemgetter, attrgetter
from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import Mapping
from pyomo.common.deprecation import deprecated, deprecation_warning, RenamedClass
from pyomo.common.formatting import StreamIndenter
from pyomo.common.gc_manager import PauseGC
from pyomo.common.log import is_debug_set
from pyomo.common.pyomo_typing import overload
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.component import (
from pyomo.core.base.enums import SortComponents, TraversalStrategy
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.set import Any
from pyomo.core.base.var import Var
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.indexed_component import (
from pyomo.opt.base import ProblemFormat, guess_format
from pyomo.opt import WriterFactory
def _prefix_dfs_iterator(self, ctype, active, sort, dedup):
    """Helper function implementing a non-recursive prefix order
        depth-first search.  That is, the parent is returned before its
        children.

        Note: this method assumes it is called ONLY by the _tree_iterator
        method, which centralizes certain error checking and
        preliminaries.
        """
    dedup.seen_data.add(id(self))
    PM = PseudoMap(self, ctype, active, sort)
    _stack = (None, (self,).__iter__())
    while _stack is not None:
        try:
            PM._block = _block = next(_stack[1])
            yield _block
            if not PM:
                continue
            _stack = (_stack, _block._component_data_itervalues(ctype, active, sort, dedup))
        except StopIteration:
            _stack = _stack[0]