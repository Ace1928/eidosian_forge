import collections
import contextlib
import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.misc import quicksort
from numba.cpython import slicing
from numba.core.errors import NumbaValueError, TypingError
from numba.core.extending import overload, overload_method, intrinsic
class SetIterInstance(object):

    def __init__(self, context, builder, iter_type, iter_val):
        self._context = context
        self._builder = builder
        self._ty = iter_type
        self._iter = context.make_helper(builder, iter_type, iter_val)
        ptr = self._context.nrt.meminfo_data(builder, self.meminfo)
        self._payload = _SetPayload(context, builder, self._ty.container, ptr)

    @classmethod
    def from_set(cls, context, builder, iter_type, set_val):
        set_inst = SetInstance(context, builder, iter_type.container, set_val)
        self = cls(context, builder, iter_type, None)
        index = context.get_constant(types.intp, 0)
        self._iter.index = cgutils.alloca_once_value(builder, index)
        self._iter.meminfo = set_inst.meminfo
        return self

    @property
    def value(self):
        return self._iter._getvalue()

    @property
    def meminfo(self):
        return self._iter.meminfo

    @property
    def index(self):
        return self._builder.load(self._iter.index)

    @index.setter
    def index(self, value):
        self._builder.store(value, self._iter.index)

    def iternext(self, result):
        index = self.index
        payload = self._payload
        one = ir.Constant(index.type, 1)
        result.set_exhausted()
        with payload._iterate(start=index) as loop:
            entry = loop.entry
            result.set_valid()
            result.yield_(entry.key)
            self.index = self._builder.add(loop.index, one)
            loop.do_break()