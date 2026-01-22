from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
class GCClearReferencesSlot(GCDependentSlot):

    def slot_code(self, scope):
        if scope.needs_tp_clear():
            return GCDependentSlot.slot_code(self, scope)
        return '0'