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
class FusedTypeNode(CBaseTypeNode):
    """
    Represents a fused type in a ctypedef statement:

        ctypedef cython.fused_type(int, long, long long) integral

    name            str                     name of this fused type
    types           [CSimpleBaseTypeNode]   is the list of types to be fused
    """
    child_attrs = []

    def analyse_declarations(self, env):
        type = self.analyse(env)
        entry = env.declare_typedef(self.name, type, self.pos)
        entry.in_cinclude = True

    def analyse(self, env, could_be_name=False):
        types = []
        for type_node in self.types:
            type = type_node.analyse_as_type(env)
            if not type:
                error(type_node.pos, 'Not a type')
                continue
            if type in types:
                error(type_node.pos, 'Type specified multiple times')
            else:
                types.append(type)
        return PyrexTypes.FusedType(types, name=self.name)