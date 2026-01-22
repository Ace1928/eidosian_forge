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
def analyse_signature(self, env):
    if self.entry.is_special:
        if self.decorators:
            error(self.pos, 'special functions of cdef classes cannot have decorators')
        self.entry.trivial_signature = len(self.args) == 1 and (not (self.star_arg or self.starstar_arg))
    elif not (self.star_arg or self.starstar_arg) and (not env.directives['always_allow_keywords'] or all([arg.pos_only for arg in self.args])):
        if self.entry.signature is TypeSlots.pyfunction_signature:
            if len(self.args) == 0:
                self.entry.signature = TypeSlots.pyfunction_noargs
            elif len(self.args) == 1:
                if self.args[0].default is None and (not self.args[0].kw_only):
                    self.entry.signature = TypeSlots.pyfunction_onearg
        elif self.entry.signature is TypeSlots.pymethod_signature:
            if len(self.args) == 1:
                self.entry.signature = TypeSlots.unaryfunc
            elif len(self.args) == 2:
                if self.args[1].default is None and (not self.args[1].kw_only):
                    self.entry.signature = TypeSlots.ibinaryfunc
    sig = self.entry.signature
    nfixed = sig.max_num_fixed_args()
    min_nfixed = sig.min_num_fixed_args()
    if sig is TypeSlots.pymethod_signature and nfixed == 1 and (len(self.args) == 0) and self.star_arg:
        sig = self.entry.signature = TypeSlots.pyfunction_signature
        self.self_in_stararg = 1
        nfixed = min_nfixed = 0
    if self.is_staticmethod and env.is_c_class_scope:
        nfixed = min_nfixed = 0
        self.self_in_stararg = True
        self.entry.signature = sig = copy.copy(sig)
        sig.fixed_arg_format = '*'
        sig.is_staticmethod = True
        sig.has_generic_args = True
    if (self.is_classmethod or self.is_staticmethod) and self.has_fused_arguments and env.is_c_class_scope:
        del self.decorator_indirection.stats[:]
    for i in range(min(nfixed, len(self.args))):
        arg = self.args[i]
        arg.is_generic = 0
        if i >= min_nfixed:
            arg.is_special_method_optional = True
        if sig.is_self_arg(i) and (not self.is_staticmethod):
            if self.is_classmethod:
                arg.is_type_arg = 1
                arg.hdr_type = arg.type = Builtin.type_type
            else:
                arg.is_self_arg = 1
                arg.hdr_type = arg.type = env.parent_type
            arg.needs_conversion = 0
        else:
            arg.hdr_type = sig.fixed_arg_type(i)
            if not arg.type.same_as(arg.hdr_type):
                if arg.hdr_type.is_pyobject and arg.type.is_pyobject:
                    arg.needs_type_test = 1
                else:
                    arg.needs_conversion = 1
    if min_nfixed > len(self.args):
        self.bad_signature()
        return
    elif nfixed < len(self.args):
        if not sig.has_generic_args:
            self.bad_signature()
        for arg in self.args:
            if arg.is_generic and (arg.type.is_extension_type or arg.type.is_builtin_type):
                arg.needs_type_test = 1
    mf = sig.method_flags()
    if mf and TypeSlots.method_varargs in mf and (not self.entry.is_special):
        if self.star_arg:
            uses_args_tuple = True
            for arg in self.args:
                if arg.is_generic and (not arg.kw_only) and (not arg.is_self_arg) and (not arg.is_type_arg):
                    uses_args_tuple = False
        else:
            uses_args_tuple = False
        if not uses_args_tuple:
            sig = self.entry.signature = sig.with_fastcall()