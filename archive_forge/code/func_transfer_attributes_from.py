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
def transfer_attributes_from(self, src):
    """Transfer user-defined attributes from src to this block

        This transfers all components and user-defined attributes from
        the block or dictionary `src` and places them on this Block.
        Components are transferred in declaration order.

        If a Component on `src` is also declared on this block as either
        a Component or attribute, the local Component or attribute is
        replaced by the incoming component.  If an attribute name on
        `src` matches a Component declared on this block, then the
        incoming attribute is passed to the local Component's
        `set_value()` method.  Attribute names appearing in this block's
        `_Block_reserved_words` set will not be transferred (although
        Components will be).

        Parameters
        ----------
        src: _BlockData or dict
            The Block or mapping that contains the new attributes to
            assign to this block.
        """
    if isinstance(src, _BlockData):
        if src is self:
            return
        p_block = self.parent_block()
        while p_block is not None:
            if p_block is src:
                raise ValueError('_BlockData.transfer_attributes_from(): Cannot set a sub-block (%s) to a parent block (%s): creates a circular hierarchy' % (self, src))
            p_block = p_block.parent_block()
        src_comp_map = dict(src.component_map().items())
        src_raw_dict = src.__dict__
        del_src_comp = src.del_component
    elif isinstance(src, Mapping):
        src_comp_map = {k: v for k, v in src.items() if isinstance(v, Component)}
        src_raw_dict = src
        del_src_comp = lambda x: None
    else:
        raise ValueError('_BlockData.transfer_attributes_from(): expected a Block or dict; received %s' % (type(src).__name__,))
    if src_comp_map:
        src_raw_dict = {k: v for k, v in src_raw_dict.items() if k not in src_comp_map}
    with self._declare_reserved_components():
        for k, v in src_comp_map.items():
            if k in self._decl:
                self.del_component(k)
            del_src_comp(k)
            self.add_component(k, v)
    for k, v in src_raw_dict.items():
        if k not in self._Block_reserved_words or not hasattr(self, k) or k in self._decl:
            setattr(self, k, v)