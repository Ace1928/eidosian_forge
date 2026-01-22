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
class AutoLinkedBinaryVar(ScalarVar):
    """A binary variable implicitly linked to its equivalent Boolean variable.

    Basic operations like setting values and fixing/unfixing this
    variable are also automatically applied to the associated Boolean
    variable.

    As this class is only intended to provide a deprecation path for
    Disjunct.indicator_var, it only supports Scalar instances and does
    not support indexing.
    """
    INTEGER_TOLERANCE = 0.001
    __autoslot_mappers__ = {'_associated_boolean': AutoSlots.weakref_mapper}

    def __init__(self, boolean_var=None):
        super().__init__(domain=Binary)
        self._associated_boolean = weakref_ref(boolean_var)

    def get_associated_boolean(self):
        return self._associated_boolean()

    def set_value(self, val, skip_validation=False, _propagate_value=True):
        super().set_value(val, skip_validation)
        if not _propagate_value:
            return
        if val is None:
            bool_val = None
        elif fabs(val - 0.5) < 0.5 - AutoLinkedBinaryVar.INTEGER_TOLERANCE:
            bool_val = None
        else:
            bool_val = bool(int(val + 0.5))
        self.get_associated_boolean().set_value(bool_val, skip_validation, _propagate_value=False)

    def fix(self, value=NOTSET, skip_validation=False):
        super().fix(value, skip_validation)
        bool_var = self.get_associated_boolean()
        if not bool_var.is_fixed():
            bool_var.fix()

    def unfix(self):
        super().unfix()
        bool_var = self.get_associated_boolean()
        if bool_var.is_fixed():
            bool_var.unfix()