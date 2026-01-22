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
class SubclassOf(object):
    """This mocks up a tuple-like interface based on subclass relationship.

    Instances of this class present a somewhat tuple-like interface for
    use in PseudoMap ctype / descend_into.  The constructor takes a
    single ctype argument.  When used with PseudoMap (through Block APIs
    like component_objects()), it will match any ctype that is a
    subclass of the reference ctype.

    This allows, for example:

        model.component_data_objects(Var, descend_into=SubclassOf(Block))
    """

    def __init__(self, *ctype):
        self.ctype = ctype
        self.__name__ = 'SubclassOf(%s)' % (','.join((x.__name__ for x in ctype)),)

    def __contains__(self, item):
        return issubclass(item, self.ctype)

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self

    def __iter__(self):
        return iter((self,))