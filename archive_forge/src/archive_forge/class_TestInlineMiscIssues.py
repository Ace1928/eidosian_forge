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
class TestInlineMiscIssues(TestCase):

    def test_issue4691(self):

        def output_factory(array, dtype):
            pass

        @overload(output_factory, inline='always')
        def ol_output_factory(array, dtype):
            if isinstance(array, types.npytypes.Array):

                def impl(array, dtype):
                    shape = array.shape[3:]
                    return np.zeros(shape, dtype=dtype)
                return impl

        @njit(nogil=True)
        def fn(array):
            out = output_factory(array, array.dtype)
            return out

        @njit(nogil=True)
        def fn2(array):
            return np.zeros(array.shape[3:], dtype=array.dtype)
        fn(np.ones((10, 20, 30, 40, 50)))
        fn2(np.ones((10, 20, 30, 40, 50)))

    def test_issue4693(self):

        @njit(inline='always')
        def inlining(array):
            if array.ndim != 1:
                raise ValueError('Invalid number of dimensions')
            return array

        @njit
        def fn(array):
            return inlining(array)
        fn(np.zeros(10))

    def test_issue5476(self):

        @njit(inline='always')
        def inlining():
            msg = 'Something happened'
            raise ValueError(msg)

        @njit
        def fn():
            return inlining()
        with self.assertRaises(ValueError) as raises:
            fn()
        self.assertIn('Something happened', str(raises.exception))

    def test_issue5792(self):

        class Dummy:

            def __init__(self, data):
                self.data = data

            def div(self, other):
                return data / other.data

        class DummyType(types.Type):

            def __init__(self, data):
                self.data = data
                super().__init__(name=f'Dummy({self.data})')

        @register_model(DummyType)
        class DummyTypeModel(models.StructModel):

            def __init__(self, dmm, fe_type):
                members = [('data', fe_type.data)]
                super().__init__(dmm, fe_type, members)
        make_attribute_wrapper(DummyType, 'data', '_data')

        @intrinsic
        def init_dummy(typingctx, data):

            def codegen(context, builder, sig, args):
                typ = sig.return_type
                data, = args
                dummy = cgutils.create_struct_proxy(typ)(context, builder)
                dummy.data = data
                if context.enable_nrt:
                    context.nrt.incref(builder, sig.args[0], data)
                return dummy._getvalue()
            ret_typ = DummyType(data)
            sig = signature(ret_typ, data)
            return (sig, codegen)

        @overload(Dummy, inline='always')
        def dummy_overload(data):

            def ctor(data):
                return init_dummy(data)
            return ctor

        @overload_method(DummyType, 'div', inline='always')
        def div_overload(self, other):

            def impl(self, other):
                return self._data / other._data
            return impl

        @njit
        def test_impl(data, other_data):
            dummy = Dummy(data)
            other = Dummy(other_data)
            return dummy.div(other)
        data = 1.0
        other_data = 2.0
        res = test_impl(data, other_data)
        self.assertEqual(res, data / other_data)

    def test_issue5824(self):
        """ Similar to the above test_issue5792, checks mutation of the inlinee
        IR is local only"""

        class CustomCompiler(CompilerBase):

            def define_pipelines(self):
                pm = DefaultPassBuilder.define_nopython_pipeline(self.state)
                pm.add_pass_after(InlineOverloads, InlineOverloads)
                pm.finalize()
                return [pm]

        def bar(x):
            ...

        @overload(bar, inline='always')
        def ol_bar(x):
            if isinstance(x, types.Integer):

                def impl(x):
                    return x + 1.3
                return impl

        @njit(pipeline_class=CustomCompiler)
        def foo(z):
            return (bar(z), bar(z))
        self.assertEqual(foo(10), (11.3, 11.3))

    @skip_parfors_unsupported
    def test_issue7380(self):

        @njit(inline='always')
        def bar(x):
            for i in range(x.size):
                x[i] += 1

        @njit(parallel=True)
        def foo(a):
            for i in prange(a.shape[0]):
                bar(a[i])
        a = np.ones((10, 10))
        foo(a)
        self.assertPreciseEqual(a, 2 * np.ones_like(a))

        @njit(parallel=True)
        def foo_bad(a):
            for i in prange(a.shape[0]):
                x = a[i]
                for i in range(x.size):
                    x[i] += 1
        with self.assertRaises(errors.UnsupportedRewriteError) as e:
            foo_bad(a)
        self.assertIn('Overwrite of parallel loop index', str(e.exception))