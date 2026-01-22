import re
from io import StringIO
import numba
from numba.core import types
from numba import jit, njit
from numba.tests.support import override_config, TestCase
import unittest
class TestTypeAnnotation(unittest.TestCase):

    def findpatloc(self, lines, pat):
        for i, ln in enumerate(lines):
            if pat in ln:
                return i
        raise ValueError("can't find {!r}".format(pat))

    def getlines(self, func):
        strbuf = StringIO()
        func.inspect_types(strbuf)
        return strbuf.getvalue().splitlines()

    def test_delete(self):

        @numba.njit
        def foo(appleorange, berrycherry):
            return appleorange + berrycherry
        foo(1, 2)
        lines = self.getlines(foo)
        sa = self.findpatloc(lines, 'appleorange = arg(0, name=appleorange)')
        sb = self.findpatloc(lines, 'berrycherry = arg(1, name=berrycherry)')
        ea = self.findpatloc(lines, 'del appleorange')
        eb = self.findpatloc(lines, 'del berrycherry')
        self.assertLess(sa, ea)
        self.assertLess(sb, eb)

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

    def test_delete_standard_lifetimes(self):
        sa, sb, cast_ret, dela, delb = self._lifetimes_impl(extend=0)
        self.assertLess(sa, dela)
        self.assertLess(sb, delb)
        self.assertLess(dela, cast_ret)
        self.assertGreater(delb, cast_ret)

    def test_delete_extended_lifetimes(self):
        sa, sb, cast_ret, dela, delb = self._lifetimes_impl(extend=1)
        self.assertLess(sa, dela)
        self.assertLess(sb, delb)
        self.assertGreater(dela, cast_ret)
        self.assertGreater(delb, cast_ret)