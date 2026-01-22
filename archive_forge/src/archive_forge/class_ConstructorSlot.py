from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
class ConstructorSlot(InternalMethodSlot):

    def __init__(self, slot_name, method=None, **kargs):
        InternalMethodSlot.__init__(self, slot_name, **kargs)
        self.method = method

    def _needs_own(self, scope):
        if scope.parent_type.base_type and (not scope.has_pyobject_attrs) and (not scope.has_memoryview_attrs) and (not scope.has_cpp_constructable_attrs) and (not (self.slot_name == 'tp_new' and scope.parent_type.vtabslot_cname)):
            entry = scope.lookup_here(self.method) if self.method else None
            if not (entry and entry.is_special):
                return False
        return True

    def _parent_slot_function(self, scope):
        parent_type_scope = scope.parent_type.base_type.scope
        if scope.parent_scope is parent_type_scope.parent_scope:
            entry = scope.parent_scope.lookup_here(scope.parent_type.base_type.name)
            if entry.visibility != 'extern':
                return self.slot_code(parent_type_scope)
        return None

    def slot_code(self, scope):
        if not self._needs_own(scope):
            slot_code = self._parent_slot_function(scope)
            return slot_code or '0'
        return InternalMethodSlot.slot_code(self, scope)

    def spec_value(self, scope):
        slot_function = self.slot_code(scope)
        if self.slot_name == 'tp_dealloc' and slot_function != scope.mangle_internal('tp_dealloc'):
            return '0'
        return slot_function

    def generate_dynamic_init_code(self, scope, code):
        if self.slot_code(scope) != '0':
            return
        base_type = scope.parent_type.base_type
        if base_type.typeptr_cname:
            src = '%s->%s' % (base_type.typeptr_cname, self.slot_name)
        elif base_type.is_extension_type and base_type.typeobj_cname:
            src = '%s.%s' % (base_type.typeobj_cname, self.slot_name)
        else:
            return
        self.generate_set_slot_code(src, scope, code)