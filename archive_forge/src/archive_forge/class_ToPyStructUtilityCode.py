from __future__ import absolute_import
import copy
import hashlib
import re
from functools import partial
from itertools import product
from Cython.Utils import cached_function
from .Code import UtilityCode, LazyUtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from .Errors import error, CannotSpecialize, performance_hint
class ToPyStructUtilityCode(object):
    requires = None

    def __init__(self, type, forward_decl, env):
        self.type = type
        self.header = 'static PyObject* %s(%s)' % (type.to_py_function, type.declaration_code('s'))
        self.forward_decl = forward_decl
        self.env = env

    def __eq__(self, other):
        return isinstance(other, ToPyStructUtilityCode) and self.header == other.header

    def __hash__(self):
        return hash(self.header)

    def get_tree(self, **kwargs):
        pass

    def put_code(self, output):
        code = output['utility_code_def']
        proto = output['utility_code_proto']
        code.putln('%s {' % self.header)
        code.putln('PyObject* res;')
        code.putln('PyObject* member;')
        code.putln('res = __Pyx_PyDict_NewPresized(%d); if (unlikely(!res)) return NULL;' % len(self.type.scope.var_entries))
        for member in self.type.scope.var_entries:
            nameconst_cname = code.get_py_string_const(member.name, identifier=True)
            code.putln('%s; if (unlikely(!member)) goto bad;' % member.type.to_py_call_code('s.%s' % member.cname, 'member', member.type))
            code.putln('if (unlikely(PyDict_SetItem(res, %s, member) < 0)) goto bad;' % nameconst_cname)
            code.putln('Py_DECREF(member);')
        code.putln('return res;')
        code.putln('bad:')
        code.putln('Py_XDECREF(member);')
        code.putln('Py_DECREF(res);')
        code.putln('return NULL;')
        code.putln('}')
        if self.forward_decl:
            proto.putln(self.type.empty_declaration_code() + ';')
        proto.putln(self.header + ';')

    def inject_tree_and_scope_into(self, module_node):
        pass