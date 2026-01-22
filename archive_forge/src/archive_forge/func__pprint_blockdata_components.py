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
def _pprint_blockdata_components(self, ostream):
    import pyomo.core.base.component_order
    items = list(pyomo.core.base.component_order.items)
    items_set = set(items)
    items_set.add(Block)
    dynamic_items = set()
    for item in self._ctypes:
        if not item in items_set:
            dynamic_items.add(item)
    items.append(Block)
    items.extend(sorted(dynamic_items, key=lambda x: x.__name__))
    indented_ostream = StreamIndenter(ostream, self._PPRINT_INDENT)
    for item in items:
        keys = sorted(self.component_map(item))
        if not keys:
            continue
        ostream.write('%d %s Declarations\n' % (len(keys), item.__name__))
        for key in keys:
            self.component(key).pprint(ostream=indented_ostream)
        ostream.write('\n')
    decl_order_keys = list(self.component_map().keys())
    ostream.write('%d Declarations: %s\n' % (len(decl_order_keys), ' '.join((str(x) for x in decl_order_keys))))