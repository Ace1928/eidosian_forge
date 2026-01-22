from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
class SyntheticSlot(InternalMethodSlot):

    def __init__(self, slot_name, user_methods, default_value, **kargs):
        InternalMethodSlot.__init__(self, slot_name, **kargs)
        self.user_methods = user_methods
        self.default_value = default_value

    def slot_code(self, scope):
        if scope.defines_any_special(self.user_methods):
            return InternalMethodSlot.slot_code(self, scope)
        else:
            return self.default_value

    def spec_value(self, scope):
        return self.slot_code(scope)