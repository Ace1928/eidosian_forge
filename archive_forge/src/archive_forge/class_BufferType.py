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
class BufferType(BaseType):
    is_buffer = 1
    writable = True
    subtypes = ['dtype']

    def __init__(self, base, dtype, ndim, mode, negative_indices, cast):
        self.base = base
        self.dtype = dtype
        self.ndim = ndim
        self.buffer_ptr_type = CPtrType(dtype)
        self.mode = mode
        self.negative_indices = negative_indices
        self.cast = cast
        self.is_numpy_buffer = self.base.name == 'ndarray'

    def can_coerce_to_pyobject(self, env):
        return True

    def can_coerce_from_pyobject(self, env):
        return True

    def as_argument_type(self):
        return self

    def specialize(self, values):
        dtype = self.dtype.specialize(values)
        if dtype is not self.dtype:
            return BufferType(self.base, dtype, self.ndim, self.mode, self.negative_indices, self.cast)
        return self

    def get_entry(self, node):
        from . import Buffer
        assert node.is_name
        return Buffer.BufferEntry(node.entry)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __repr__(self):
        return '<BufferType %r>' % self.base

    def __str__(self):
        cast_str = ''
        if self.cast:
            cast_str = ',cast=True'
        return '%s[%s,ndim=%d%s]' % (self.base, self.dtype, self.ndim, cast_str)

    def assignable_from(self, other_type):
        if other_type.is_buffer:
            return self.same_as(other_type, compare_base=False) and self.base.assignable_from(other_type.base)
        return self.base.assignable_from(other_type)

    def same_as(self, other_type, compare_base=True):
        if not other_type.is_buffer:
            return other_type.same_as(self.base)
        return self.dtype.same_as(other_type.dtype) and self.ndim == other_type.ndim and (self.mode == other_type.mode) and (self.cast == other_type.cast) and (not compare_base or self.base.same_as(other_type.base))