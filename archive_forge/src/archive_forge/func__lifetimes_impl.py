import re
from io import StringIO
import numba
from numba.core import types
from numba import jit, njit
from numba.tests.support import override_config, TestCase
import unittest
def _lifetimes_impl(self, extend):
    with override_config('EXTEND_VARIABLE_LIFETIMES', extend):

        @njit
        def foo(a):
            b = a
            return b
        x = 10
        b = foo(x)
        self.assertEqual(b, x)
    lines = self.getlines(foo)
    sa = self.findpatloc(lines, 'a = arg(0, name=a)')
    sb = self.findpatloc(lines, 'b = a')
    cast_ret = self.findpatloc(lines, 'cast(value=b)')
    dela = self.findpatloc(lines, 'del a')
    delb = self.findpatloc(lines, 'del b')
    return (sa, sb, cast_ret, dela, delb)