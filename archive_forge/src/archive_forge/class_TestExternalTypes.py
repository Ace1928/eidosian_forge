import gc
from io import StringIO
import numpy as np
from numba import njit, vectorize
from numba import typeof
from numba.core import utils, types, typing, ir, compiler, cpu, cgutils
from numba.core.compiler import Compiler, Flags
from numba.core.registry import cpu_target
from numba.tests.support import (MemoryLeakMixin, TestCase, temp_directory,
from numba.extending import (
import operator
import textwrap
import unittest
class TestExternalTypes(MemoryLeakMixin, unittest.TestCase):
    """ Tests RewriteArrayExprs with external (user defined) types,
    see #5157"""
    source_lines = textwrap.dedent("\n        from numba.core import types\n\n        class FooType(types.Type):\n            def __init__(self):\n                super(FooType, self).__init__(name='Foo')\n        ")

    def make_foo_type(self, FooType):

        class Foo(object):

            def __init__(self, value):
                self.value = value

        @register_model(FooType)
        class FooModel(models.StructModel):

            def __init__(self, dmm, fe_type):
                members = [('value', types.intp)]
                models.StructModel.__init__(self, dmm, fe_type, members)
        make_attribute_wrapper(FooType, 'value', 'value')

        @type_callable(Foo)
        def type_foo(context):

            def typer(value):
                return FooType()
            return typer

        @lower_builtin(Foo, types.intp)
        def impl_foo(context, builder, sig, args):
            typ = sig.return_type
            [value] = args
            foo = cgutils.create_struct_proxy(typ)(context, builder)
            foo.value = value
            return foo._getvalue()

        @typeof_impl.register(Foo)
        def typeof_foo(val, c):
            return FooType()
        return (Foo, FooType)

    def test_external_type(self):
        with create_temp_module(self.source_lines) as test_module:
            Foo, FooType = self.make_foo_type(test_module.FooType)

            @overload(operator.add)
            def overload_foo_add(lhs, rhs):
                if isinstance(lhs, FooType) and isinstance(rhs, types.Array):

                    def imp(lhs, rhs):
                        return np.array([lhs.value, rhs[0]])
                    return imp

            @overload(operator.add)
            def overload_foo_add(lhs, rhs):
                if isinstance(lhs, FooType) and isinstance(rhs, FooType):

                    def imp(lhs, rhs):
                        return np.array([lhs.value, rhs.value])
                    return imp

            @overload(operator.neg)
            def overload_foo_neg(x):
                if isinstance(x, FooType):

                    def imp(x):
                        return np.array([-x.value])
                    return imp

            @njit
            def arr_expr_sum1(x, y):
                return Foo(x) + np.array([y])

            @njit
            def arr_expr_sum2(x, y):
                return Foo(x) + Foo(y)

            @njit
            def arr_expr_neg(x):
                return -Foo(x)
            np.testing.assert_array_equal(arr_expr_sum1(0, 1), np.array([0, 1]))
            np.testing.assert_array_equal(arr_expr_sum2(2, 3), np.array([2, 3]))
            np.testing.assert_array_equal(arr_expr_neg(4), np.array([-4]))