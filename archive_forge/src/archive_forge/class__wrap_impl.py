from collections import defaultdict
import copy
import sys
from itertools import permutations, takewhile
from contextlib import contextmanager
from functools import cached_property
from llvmlite import ir as llvmir
from llvmlite.ir import Constant
import llvmlite.binding as ll
from numba.core import types, utils, datamodel, debuginfo, funcdesc, config, cgutils, imputils
from numba.core import event, errors, targetconfig
from numba import _dynfunc, _helperlib
from numba.core.compiler_lock import global_compiler_lock
from numba.core.pythonapi import PythonAPI
from numba.core.imputils import (user_function, user_generator,
from numba.cpython import builtins
class _wrap_impl(object):
    """
    A wrapper object to call an implementation function with some predefined
    (context, signature) arguments.
    The wrapper also forwards attribute queries, which is important.
    """

    def __init__(self, imp, context, sig):
        self._callable = _wrap_missing_loc(imp)
        self._imp = self._callable()
        self._context = context
        self._sig = sig

    def __call__(self, builder, args, loc=None):
        res = self._imp(self._context, builder, self._sig, args, loc=loc)
        self._context.add_linking_libs(getattr(self, 'libs', ()))
        return res

    def __getattr__(self, item):
        return getattr(self._imp, item)

    def __repr__(self):
        return '<wrapped %s>' % repr(self._callable)