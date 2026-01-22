from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
class SlotDescriptor(object):

    def __init__(self, slot_name, dynamic=False, inherited=False, py3=True, py2=True, ifdef=None, is_binop=False, used_ifdef=None):
        self.slot_name = slot_name
        self.is_initialised_dynamically = dynamic
        self.is_inherited = inherited
        self.ifdef = ifdef
        self.used_ifdef = used_ifdef
        self.py3 = py3
        self.py2 = py2
        self.is_binop = is_binop

    def slot_code(self, scope):
        raise NotImplementedError()

    def spec_value(self, scope):
        return self.slot_code(scope)

    def preprocessor_guard_code(self):
        ifdef = self.ifdef
        py2 = self.py2
        py3 = self.py3
        guard = None
        if ifdef:
            guard = '#if %s' % ifdef
        elif not py3 or py3 == '<RESERVED>':
            guard = '#if PY_MAJOR_VERSION < 3'
        elif not py2:
            guard = '#if PY_MAJOR_VERSION >= 3'
        return guard

    def generate_spec(self, scope, code):
        if self.is_initialised_dynamically:
            return
        value = self.spec_value(scope)
        if value == '0':
            return
        preprocessor_guard = self.preprocessor_guard_code()
        if not preprocessor_guard:
            if self.py3 and self.slot_name.startswith('bf_'):
                preprocessor_guard = '#if defined(Py_%s)' % self.slot_name
        if preprocessor_guard:
            code.putln(preprocessor_guard)
        code.putln('{Py_%s, (void *)%s},' % (self.slot_name, value))
        if preprocessor_guard:
            code.putln('#endif')

    def generate(self, scope, code):
        preprocessor_guard = self.preprocessor_guard_code()
        if preprocessor_guard:
            code.putln(preprocessor_guard)
        end_pypy_guard = False
        if self.is_initialised_dynamically:
            value = '0'
        else:
            value = self.slot_code(scope)
            if value == '0' and self.is_inherited:
                inherited_value = value
                current_scope = scope
                while inherited_value == '0' and current_scope.parent_type and current_scope.parent_type.base_type and current_scope.parent_type.base_type.scope:
                    current_scope = current_scope.parent_type.base_type.scope
                    inherited_value = self.slot_code(current_scope)
                if inherited_value != '0':
                    is_buffer_slot = int(self.slot_name in ('bf_getbuffer', 'bf_releasebuffer'))
                    code.putln('#if CYTHON_COMPILING_IN_PYPY || %d' % is_buffer_slot)
                    code.putln('%s, /*%s*/' % (inherited_value, self.slot_name))
                    code.putln('#else')
                    end_pypy_guard = True
        if self.used_ifdef:
            code.putln('#if %s' % self.used_ifdef)
        code.putln('%s, /*%s*/' % (value, self.slot_name))
        if self.used_ifdef:
            code.putln('#else')
            code.putln('NULL, /*%s*/' % self.slot_name)
            code.putln('#endif')
        if end_pypy_guard:
            code.putln('#endif')
        if self.py3 == '<RESERVED>':
            code.putln('#else')
            code.putln('0, /*reserved*/')
        if preprocessor_guard:
            code.putln('#endif')

    def generate_dynamic_init_code(self, scope, code):
        if self.is_initialised_dynamically:
            self.generate_set_slot_code(self.slot_code(scope), scope, code)

    def generate_set_slot_code(self, value, scope, code):
        if value == '0':
            return
        if scope.parent_type.typeptr_cname:
            target = '%s->%s' % (scope.parent_type.typeptr_cname, self.slot_name)
        else:
            assert scope.parent_type.typeobj_cname
            target = '%s.%s' % (scope.parent_type.typeobj_cname, self.slot_name)
        code.putln('%s = %s;' % (target, value))