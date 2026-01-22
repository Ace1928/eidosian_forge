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
class CheckAnalysers(type):
    """Metaclass to check that type analysis functions return a node.
    """
    methods = frozenset({'analyse_types', 'analyse_expressions', 'analyse_target_types'})

    def __new__(cls, name, bases, attrs):
        from types import FunctionType

        def check(name, func):

            def call(*args, **kwargs):
                retval = func(*args, **kwargs)
                if retval is None:
                    print('%s %s %s' % (name, args, kwargs))
                return retval
            return call
        attrs = dict(attrs)
        for mname, m in attrs.items():
            if isinstance(m, FunctionType) and mname in cls.methods:
                attrs[mname] = check(mname, m)
        return super(CheckAnalysers, cls).__new__(cls, name, bases, attrs)