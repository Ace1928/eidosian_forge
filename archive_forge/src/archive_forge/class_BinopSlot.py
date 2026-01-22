from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
class BinopSlot(SyntheticSlot):

    def __init__(self, signature, slot_name, left_method, method_name_to_slot, **kargs):
        assert left_method.startswith('__')
        right_method = '__r' + left_method[2:]
        SyntheticSlot.__init__(self, slot_name, [left_method, right_method], '0', is_binop=True, **kargs)
        self.left_slot = MethodSlot(signature, '', left_method, method_name_to_slot, **kargs)
        self.right_slot = MethodSlot(signature, '', right_method, method_name_to_slot, **kargs)