from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
class BaseClassSlot(SlotDescriptor):

    def __init__(self, name):
        SlotDescriptor.__init__(self, name, dynamic=True)

    def generate_dynamic_init_code(self, scope, code):
        base_type = scope.parent_type.base_type
        if base_type:
            code.putln('%s->%s = %s;' % (scope.parent_type.typeptr_cname, self.slot_name, base_type.typeptr_cname))