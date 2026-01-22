import multiprocessing
import platform
import threading
import pickle
import weakref
from itertools import chain
from io import StringIO
import numpy as np
from numba import njit, jit, typeof, vectorize
from numba.core import types, errors
from numba import _dispatcher
from numba.tests.support import TestCase, captured_stdout
from numba.np.numpy_support import as_dtype
from numba.core.dispatcher import Dispatcher
from numba.extending import overload
from numba.tests.support import needs_lapack, SerialMixin
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
import unittest
class TestDispatcherMethods(TestCase):

    def test_recompile(self):
        closure = 1

        @jit
        def foo(x):
            return x + closure
        self.assertPreciseEqual(foo(1), 2)
        self.assertPreciseEqual(foo(1.5), 2.5)
        self.assertEqual(len(foo.signatures), 2)
        closure = 2
        self.assertPreciseEqual(foo(1), 2)
        foo.recompile()
        self.assertEqual(len(foo.signatures), 2)
        self.assertPreciseEqual(foo(1), 3)
        self.assertPreciseEqual(foo(1.5), 3.5)

    def test_recompile_signatures(self):
        closure = 1

        @jit('int32(int32)')
        def foo(x):
            return x + closure
        self.assertPreciseEqual(foo(1), 2)
        self.assertPreciseEqual(foo(1.5), 2)
        closure = 2
        self.assertPreciseEqual(foo(1), 2)
        foo.recompile()
        self.assertPreciseEqual(foo(1), 3)
        self.assertPreciseEqual(foo(1.5), 3)

    def test_inspect_llvm(self):

        @jit
        def foo(explicit_arg1, explicit_arg2):
            return explicit_arg1 + explicit_arg2
        foo(1, 1)
        foo(1.0, 1)
        foo(1.0, 1.0)
        llvms = foo.inspect_llvm()
        self.assertEqual(len(llvms), 3)
        for llvm_bc in llvms.values():
            self.assertIn('foo', llvm_bc)
            self.assertIn('explicit_arg1', llvm_bc)
            self.assertIn('explicit_arg2', llvm_bc)

    def test_inspect_asm(self):

        @jit
        def foo(explicit_arg1, explicit_arg2):
            return explicit_arg1 + explicit_arg2
        foo(1, 1)
        foo(1.0, 1)
        foo(1.0, 1.0)
        asms = foo.inspect_asm()
        self.assertEqual(len(asms), 3)
        for asm in asms.values():
            self.assertTrue('foo' in asm)

    def _check_cfg_display(self, cfg, wrapper=''):
        if wrapper:
            wrapper = '{}{}'.format(len(wrapper), wrapper)
        module_name = __name__.split('.', 1)[0]
        module_len = len(module_name)
        prefix = '^digraph "CFG for \\\'_ZN{}{}{}'.format(wrapper, module_len, module_name)
        self.assertRegex(str(cfg), prefix)
        self.assertTrue(callable(cfg.display))

    def test_inspect_cfg(self):

        @jit
        def foo(the_array):
            return the_array.sum()
        a1 = np.ones(1)
        a2 = np.ones((1, 1))
        a3 = np.ones((1, 1, 1))
        foo(a1)
        foo(a2)
        foo(a3)
        cfgs = foo.inspect_cfg()
        self.assertEqual(len(cfgs), 3)
        [s1, s2, s3] = cfgs.keys()
        self.assertEqual(set([s1, s2, s3]), set(map(lambda x: (typeof(x),), [a1, a2, a3])))
        for cfg in cfgs.values():
            self._check_cfg_display(cfg)
        self.assertEqual(len(list(cfgs.values())), 3)
        cfg = foo.inspect_cfg(signature=foo.signatures[0])
        self._check_cfg_display(cfg)

    def test_inspect_cfg_with_python_wrapper(self):

        @jit
        def foo(the_array):
            return the_array.sum()
        a1 = np.ones(1)
        a2 = np.ones((1, 1))
        a3 = np.ones((1, 1, 1))
        foo(a1)
        foo(a2)
        foo(a3)
        cfg = foo.inspect_cfg(signature=foo.signatures[0], show_wrapper='python')
        self._check_cfg_display(cfg, wrapper='cpython')

    def test_inspect_types(self):

        @jit
        def foo(a, b):
            return a + b
        foo(1, 2)
        foo.inspect_types(StringIO())
        expected = str(foo.overloads[foo.signatures[0]].type_annotation)
        with captured_stdout() as out:
            foo.inspect_types()
        assert expected in out.getvalue()

    def test_inspect_types_with_signature(self):

        @jit
        def foo(a):
            return a + 1
        foo(1)
        foo(1.0)
        with captured_stdout() as total:
            foo.inspect_types()
        with captured_stdout() as first:
            foo.inspect_types(signature=foo.signatures[0])
        with captured_stdout() as second:
            foo.inspect_types(signature=foo.signatures[1])
        self.assertEqual(total.getvalue(), first.getvalue() + second.getvalue())

    @unittest.skipIf(jinja2 is None, "please install the 'jinja2' package")
    @unittest.skipIf(pygments is None, "please install the 'pygments' package")
    def test_inspect_types_pretty(self):

        @jit
        def foo(a, b):
            return a + b
        foo(1, 2)
        with captured_stdout():
            ann = foo.inspect_types(pretty=True)
        for k, v in ann.ann.items():
            span_found = False
            for line in v['pygments_lines']:
                if 'span' in line[2]:
                    span_found = True
            self.assertTrue(span_found)
        with self.assertRaises(ValueError) as raises:
            foo.inspect_types(file=StringIO(), pretty=True)
        self.assertIn('`file` must be None if `pretty=True`', str(raises.exception))

    def test_get_annotation_info(self):

        @jit
        def foo(a):
            return a + 1
        foo(1)
        foo(1.3)
        expected = dict(chain.from_iterable((foo.get_annotation_info(i).items() for i in foo.signatures)))
        result = foo.get_annotation_info()
        self.assertEqual(expected, result)

    def test_issue_with_array_layout_conflict(self):
        """
        This test an issue with the dispatcher when an array that is both
        C and F contiguous is supplied as the first signature.
        The dispatcher checks for F contiguous first but the compiler checks
        for C contiguous first. This results in an C contiguous code inserted
        as F contiguous function.
        """

        def pyfunc(A, i, j):
            return A[i, j]
        cfunc = jit(pyfunc)
        ary_c_and_f = np.array([[1.0]])
        ary_c = np.array([[0.0, 1.0], [2.0, 3.0]], order='C')
        ary_f = np.array([[0.0, 1.0], [2.0, 3.0]], order='F')
        exp_c = pyfunc(ary_c, 1, 0)
        exp_f = pyfunc(ary_f, 1, 0)
        self.assertEqual(1.0, cfunc(ary_c_and_f, 0, 0))
        got_c = cfunc(ary_c, 1, 0)
        got_f = cfunc(ary_f, 1, 0)
        self.assertEqual(exp_c, got_c)
        self.assertEqual(exp_f, got_f)