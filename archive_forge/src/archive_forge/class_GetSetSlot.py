from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
class GetSetSlot(SlotDescriptor):

    def slot_code(self, scope):
        if scope.property_entries:
            return scope.getset_table_cname
        else:
            return '0'