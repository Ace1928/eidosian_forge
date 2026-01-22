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
class CConstOrVolatileTypeNode(CBaseTypeNode):
    child_attrs = ['base_type']

    def analyse(self, env, could_be_name=False):
        base = self.base_type.analyse(env, could_be_name)
        if base.is_pyobject:
            error(self.pos, 'Const/volatile base type cannot be a Python object')
        return PyrexTypes.c_const_or_volatile_type(base, self.is_const, self.is_volatile)