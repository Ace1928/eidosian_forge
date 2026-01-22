from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
def _analyse_template_types(self, env, base_type):
    require_optional_types = base_type.python_type_constructor_name == 'typing.Optional'
    require_python_types = base_type.python_type_constructor_name == 'dataclasses.ClassVar'
    in_c_type_context = env.in_c_type_context and (not require_python_types)
    template_types = []
    for template_node in self.positional_args:
        with env.new_c_type_context(in_c_type_context or isinstance(template_node, CBaseTypeNode)):
            ttype = template_node.analyse_as_type(env)
        if ttype is None:
            if base_type.is_cpp_class:
                error(template_node.pos, 'unknown type in template argument')
                ttype = error_type
        elif require_python_types and (not ttype.is_pyobject) or (require_optional_types and (not ttype.can_be_optional())):
            if ttype.equivalent_type and (not template_node.as_cython_attribute()):
                ttype = ttype.equivalent_type
            else:
                error(template_node.pos, '%s[...] cannot be applied to type %s' % (base_type.python_type_constructor_name, ttype))
                ttype = error_type
        template_types.append(ttype)
    return template_types