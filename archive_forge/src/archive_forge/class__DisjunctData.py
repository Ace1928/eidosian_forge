import logging
import sys
import types
from math import fabs
from weakref import ref as weakref_ref
from pyomo.common.autoslots import AutoSlots
from pyomo.common.deprecation import deprecation_warning, RenamedClass
from pyomo.common.errors import PyomoException
from pyomo.common.log import is_debug_set
from pyomo.common.numeric_types import native_logical_types, native_types
from pyomo.common.modeling import unique_component_name, NOTSET
from pyomo.common.timing import ConstructionTimer
from pyomo.core import (
from pyomo.core.base.component import (
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.block import _BlockData
from pyomo.core.base.misc import apply_indexed_rule
from pyomo.core.base.indexed_component import ActiveIndexedComponent
from pyomo.core.expr.expr_common import ExpressionType
class _DisjunctData(_BlockData):
    __autoslot_mappers__ = {'_transformation_block': AutoSlots.weakref_mapper}
    _Block_reserved_words = set()

    @property
    def transformation_block(self):
        return None if self._transformation_block is None else self._transformation_block()

    def __init__(self, component):
        _BlockData.__init__(self, component)
        with self._declare_reserved_components():
            self.indicator_var = AutoLinkedBooleanVar()
            self.binary_indicator_var = AutoLinkedBinaryVar(self.indicator_var)
        self.indicator_var.associate_binary_var(self.binary_indicator_var)
        self._transformation_block = None

    def activate(self):
        super(_DisjunctData, self).activate()
        self.indicator_var.unfix()

    def deactivate(self):
        super(_DisjunctData, self).deactivate()
        self.indicator_var.fix(False)

    def _deactivate_without_fixing_indicator(self):
        super(_DisjunctData, self).deactivate()

    def _activate_without_unfixing_indicator(self):
        super(_DisjunctData, self).activate()