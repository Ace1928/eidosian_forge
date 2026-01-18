import operator
import warnings
from itertools import product
import numpy as np
from numba import njit, typeof, literally, prange
from numba.core import types, ir, ir_utils, cgutils, errors, utils
from numba.core.extending import (
from numba.core.cpu import InlineOptions
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.typed_passes import InlineOverloads
from numba.core.typing import signature
from numba.tests.support import (TestCase, unittest,
def test_overload_method_default_args_always(self):
    Dummy, DummyType = self.make_dummy_type()

    @overload_method(DummyType, 'inline_method', inline='always')
    def _get_inlined_method(obj, val=None, val2=None):

        def get(obj, val=None, val2=None):
            return ('THIS IS INLINED', val, val2)
        return get

    def foo(obj):
        return (obj.inline_method(123), obj.inline_method(val2=321))
    self.check_method(test_impl=foo, args=[Dummy()], expected=(('THIS IS INLINED', 123, None), ('THIS IS INLINED', None, 321)), block_count=1)