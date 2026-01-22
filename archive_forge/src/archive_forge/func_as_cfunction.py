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
def as_cfunction(self, cfunc=None, scope=None, overridable=True, returns=None, except_val=None, has_explicit_exc_clause=False, modifiers=None, nogil=False, with_gil=False):
    if self.star_arg:
        error(self.star_arg.pos, 'cdef function cannot have star argument')
    if self.starstar_arg:
        error(self.starstar_arg.pos, 'cdef function cannot have starstar argument')
    exception_value, exception_check = except_val or (None, False)
    nogil = nogil or with_gil
    if cfunc is None:
        cfunc_args = []
        for formal_arg in self.args:
            name_declarator, type = formal_arg.analyse(scope, nonempty=1)
            cfunc_args.append(PyrexTypes.CFuncTypeArg(name=name_declarator.name, cname=None, annotation=formal_arg.annotation, type=py_object_type, pos=formal_arg.pos))
        cfunc_type = PyrexTypes.CFuncType(return_type=py_object_type, args=cfunc_args, has_varargs=False, exception_value=None, exception_check=exception_check, nogil=nogil, with_gil=with_gil, is_overridable=overridable)
        cfunc = CVarDefNode(self.pos, type=cfunc_type)
    else:
        if scope is None:
            scope = cfunc.scope
        cfunc_type = cfunc.type
        if len(self.args) != len(cfunc_type.args) or cfunc_type.has_varargs:
            error(self.pos, 'wrong number of arguments')
            error(cfunc.pos, 'previous declaration here')
        for i, (formal_arg, type_arg) in enumerate(zip(self.args, cfunc_type.args)):
            name_declarator, type = formal_arg.analyse(scope, nonempty=1, is_self_arg=i == 0 and scope.is_c_class_scope)
            if type is None or type is PyrexTypes.py_object_type:
                formal_arg.type = type_arg.type
                formal_arg.name_declarator = name_declarator
    if exception_value is None and cfunc_type.exception_value is not None:
        from .ExprNodes import ConstNode
        exception_value = ConstNode(self.pos, value=cfunc_type.exception_value, type=cfunc_type.return_type)
    declarator = CFuncDeclaratorNode(self.pos, base=CNameDeclaratorNode(self.pos, name=self.name, cname=None), args=self.args, has_varargs=False, exception_check=cfunc_type.exception_check, exception_value=exception_value, has_explicit_exc_clause=has_explicit_exc_clause, with_gil=cfunc_type.with_gil, nogil=cfunc_type.nogil)
    return CFuncDefNode(self.pos, modifiers=modifiers or [], base_type=CAnalysedBaseTypeNode(self.pos, type=cfunc_type.return_type), declarator=declarator, body=self.body, doc=self.doc, overridable=cfunc_type.is_overridable, type=cfunc_type, with_gil=cfunc_type.with_gil, nogil=cfunc_type.nogil, visibility='private', api=False, directive_locals=getattr(cfunc, 'directive_locals', {}), directive_returns=returns)