import cmath
import contextlib
from collections import defaultdict
import enum
import gc
import math
import platform
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import io
import ctypes
import multiprocessing as mp
import warnings
import traceback
from contextlib import contextmanager
import uuid
import importlib
import types as pytypes
from functools import cached_property
import numpy as np
from numba import testing, types
from numba.core import errors, typing, utils, config, cpu
from numba.core.typing import cffi_utils
from numba.core.compiler import (compile_extra, Flags,
from numba.core.typed_passes import IRLegalization
from numba.core.untyped_passes import PreserveIR
import unittest
from numba.core.runtime import rtsys
from numba.np import numpy_support
from numba.core.runtime import _nrt_python as _nrt
from numba.core.extending import (
from numba.core.datamodel.models import OpaqueModel
class TestCase(unittest.TestCase):
    longMessage = True

    @cached_property
    def random(self):
        return np.random.RandomState(42)

    def reset_module_warnings(self, module):
        """
        Reset the warnings registry of a module.  This can be necessary
        as the warnings module is buggy in that regard.
        See http://bugs.python.org/issue4180
        """
        if isinstance(module, str):
            module = sys.modules[module]
        try:
            del module.__warningregistry__
        except AttributeError:
            pass

    @contextlib.contextmanager
    def assertTypingError(self):
        """
        A context manager that asserts the enclosed code block fails
        compiling in nopython mode.
        """
        _accepted_errors = (errors.LoweringError, errors.TypingError, TypeError, NotImplementedError)
        with self.assertRaises(_accepted_errors) as cm:
            yield cm

    @contextlib.contextmanager
    def assertRefCount(self, *objects):
        """
        A context manager that asserts the given objects have the
        same reference counts before and after executing the
        enclosed block.
        """
        old_refcounts = [sys.getrefcount(x) for x in objects]
        yield
        gc.collect()
        new_refcounts = [sys.getrefcount(x) for x in objects]
        for old, new, obj in zip(old_refcounts, new_refcounts, objects):
            if old != new:
                self.fail('Refcount changed from %d to %d for object: %r' % (old, new, obj))

    def assertRefCountEqual(self, *objects):
        gc.collect()
        rc = [sys.getrefcount(x) for x in objects]
        rc_0 = rc[0]
        for i in range(len(objects))[1:]:
            rc_i = rc[i]
            if rc_0 != rc_i:
                self.fail(f'Refcount for objects does not match. #0({rc_0}) != #{i}({rc_i}) does not match.')

    @contextlib.contextmanager
    def assertNoNRTLeak(self):
        """
        A context manager that asserts no NRT leak was created during
        the execution of the enclosed block.
        """
        old = rtsys.get_allocation_stats()
        yield
        new = rtsys.get_allocation_stats()
        total_alloc = new.alloc - old.alloc
        total_free = new.free - old.free
        total_mi_alloc = new.mi_alloc - old.mi_alloc
        total_mi_free = new.mi_free - old.mi_free
        self.assertEqual(total_alloc, total_free, 'number of data allocs != number of data frees')
        self.assertEqual(total_mi_alloc, total_mi_free, 'number of meminfo allocs != number of meminfo frees')
    _bool_types = (bool, np.bool_)
    _exact_typesets = [_bool_types, (int,), (str,), (np.integer,), (bytes, np.bytes_)]
    _approx_typesets = [(float,), (complex,), np.inexact]
    _sequence_typesets = [(tuple, list)]
    _float_types = (float, np.floating)
    _complex_types = (complex, np.complexfloating)

    def _detect_family(self, numeric_object):
        """
        This function returns a string description of the type family
        that the object in question belongs to.  Possible return values
        are: "exact", "complex", "approximate", "sequence", and "unknown"
        """
        if isinstance(numeric_object, np.ndarray):
            return 'ndarray'
        if isinstance(numeric_object, enum.Enum):
            return 'enum'
        for tp in self._sequence_typesets:
            if isinstance(numeric_object, tp):
                return 'sequence'
        for tp in self._exact_typesets:
            if isinstance(numeric_object, tp):
                return 'exact'
        for tp in self._complex_types:
            if isinstance(numeric_object, tp):
                return 'complex'
        for tp in self._approx_typesets:
            if isinstance(numeric_object, tp):
                return 'approximate'
        return 'unknown'

    def _fix_dtype(self, dtype):
        """
        Fix the given *dtype* for comparison.
        """
        if sys.platform == 'win32' and sys.maxsize > 2 ** 32 and (dtype == np.dtype('int32')):
            return np.dtype('int64')
        else:
            return dtype

    def _fix_strides(self, arr):
        """
        Return the strides of the given array, fixed for comparison.
        Strides for 0- or 1-sized dimensions are ignored.
        """
        if arr.size == 0:
            return [0] * arr.ndim
        else:
            return [stride / arr.itemsize for stride, shape in zip(arr.strides, arr.shape) if shape > 1]

    def assertStridesEqual(self, first, second):
        """
        Test that two arrays have the same shape and strides.
        """
        self.assertEqual(first.shape, second.shape, 'shapes differ')
        self.assertEqual(first.itemsize, second.itemsize, 'itemsizes differ')
        self.assertEqual(self._fix_strides(first), self._fix_strides(second), 'strides differ')

    def assertPreciseEqual(self, first, second, prec='exact', ulps=1, msg=None, ignore_sign_on_zero=False, abs_tol=None):
        """
        Versatile equality testing function with more built-in checks than
        standard assertEqual().

        For arrays, test that layout, dtype, shape are identical, and
        recursively call assertPreciseEqual() on the contents.

        For other sequences, recursively call assertPreciseEqual() on
        the contents.

        For scalars, test that two scalars or have similar types and are
        equal up to a computed precision.
        If the scalars are instances of exact types or if *prec* is
        'exact', they are compared exactly.
        If the scalars are instances of inexact types (float, complex)
        and *prec* is not 'exact', then the number of significant bits
        is computed according to the value of *prec*: 53 bits if *prec*
        is 'double', 24 bits if *prec* is single.  This number of bits
        can be lowered by raising the *ulps* value.
        ignore_sign_on_zero can be set to True if zeros are to be considered
        equal regardless of their sign bit.
        abs_tol if this is set to a float value its value is used in the
        following. If, however, this is set to the string "eps" then machine
        precision of the type(first) is used in the following instead. This
        kwarg is used to check if the absolute difference in value between first
        and second is less than the value set, if so the numbers being compared
        are considered equal. (This is to handle small numbers typically of
        magnitude less than machine precision).

        Any value of *prec* other than 'exact', 'single' or 'double'
        will raise an error.
        """
        try:
            self._assertPreciseEqual(first, second, prec, ulps, msg, ignore_sign_on_zero, abs_tol)
        except AssertionError as exc:
            failure_msg = str(exc)
        else:
            return
        self.fail('when comparing %s and %s: %s' % (first, second, failure_msg))

    def _assertPreciseEqual(self, first, second, prec='exact', ulps=1, msg=None, ignore_sign_on_zero=False, abs_tol=None):
        """Recursive workhorse for assertPreciseEqual()."""

        def _assertNumberEqual(first, second, delta=None):
            if delta is None or first == second == 0.0 or math.isinf(first) or math.isinf(second):
                self.assertEqual(first, second, msg=msg)
                if not ignore_sign_on_zero:
                    try:
                        if math.copysign(1, first) != math.copysign(1, second):
                            self.fail(self._formatMessage(msg, '%s != %s' % (first, second)))
                    except TypeError:
                        pass
            else:
                self.assertAlmostEqual(first, second, delta=delta, msg=msg)
        first_family = self._detect_family(first)
        second_family = self._detect_family(second)
        assertion_message = 'Type Family mismatch. (%s != %s)' % (first_family, second_family)
        if msg:
            assertion_message += ': %s' % (msg,)
        self.assertEqual(first_family, second_family, msg=assertion_message)
        compare_family = first_family
        if compare_family == 'ndarray':
            dtype = self._fix_dtype(first.dtype)
            self.assertEqual(dtype, self._fix_dtype(second.dtype))
            self.assertEqual(first.ndim, second.ndim, 'different number of dimensions')
            self.assertEqual(first.shape, second.shape, 'different shapes')
            self.assertEqual(first.flags.writeable, second.flags.writeable, 'different mutability')
            self.assertEqual(self._fix_strides(first), self._fix_strides(second), 'different strides')
            if first.dtype != dtype:
                first = first.astype(dtype)
            if second.dtype != dtype:
                second = second.astype(dtype)
            for a, b in zip(first.flat, second.flat):
                self._assertPreciseEqual(a, b, prec, ulps, msg, ignore_sign_on_zero, abs_tol)
            return
        elif compare_family == 'sequence':
            self.assertEqual(len(first), len(second), msg=msg)
            for a, b in zip(first, second):
                self._assertPreciseEqual(a, b, prec, ulps, msg, ignore_sign_on_zero, abs_tol)
            return
        elif compare_family == 'exact':
            exact_comparison = True
        elif compare_family in ['complex', 'approximate']:
            exact_comparison = False
        elif compare_family == 'enum':
            self.assertIs(first.__class__, second.__class__)
            self._assertPreciseEqual(first.value, second.value, prec, ulps, msg, ignore_sign_on_zero, abs_tol)
            return
        elif compare_family == 'unknown':
            self.assertIs(first.__class__, second.__class__)
            exact_comparison = True
        else:
            assert 0, 'unexpected family'
        if hasattr(first, 'dtype') and hasattr(second, 'dtype'):
            self.assertEqual(first.dtype, second.dtype)
        if isinstance(first, self._bool_types) != isinstance(second, self._bool_types):
            assertion_message = 'Mismatching return types (%s vs. %s)' % (first.__class__, second.__class__)
            if msg:
                assertion_message += ': %s' % (msg,)
            self.fail(assertion_message)
        try:
            if cmath.isnan(first) and cmath.isnan(second):
                return
        except TypeError:
            pass
        if abs_tol is not None:
            if abs_tol == 'eps':
                rtol = np.finfo(type(first)).eps
            elif isinstance(abs_tol, float):
                rtol = abs_tol
            else:
                raise ValueError('abs_tol is not "eps" or a float, found %s' % abs_tol)
            if abs(first - second) < rtol:
                return
        exact_comparison = exact_comparison or prec == 'exact'
        if not exact_comparison and prec != 'exact':
            if prec == 'single':
                bits = 24
            elif prec == 'double':
                bits = 53
            else:
                raise ValueError('unsupported precision %r' % (prec,))
            k = 2 ** (ulps - bits - 1)
            delta = k * (abs(first) + abs(second))
        else:
            delta = None
        if isinstance(first, self._complex_types):
            _assertNumberEqual(first.real, second.real, delta)
            _assertNumberEqual(first.imag, second.imag, delta)
        elif isinstance(first, (np.timedelta64, np.datetime64)):
            if np.isnat(first):
                self.assertEqual(np.isnat(first), np.isnat(second))
            else:
                _assertNumberEqual(first, second, delta)
        else:
            _assertNumberEqual(first, second, delta)

    def subprocess_test_runner(self, test_module, test_class=None, test_name=None, envvars=None, timeout=60):
        """
        Runs named unit test(s) as specified in the arguments as:
        test_module.test_class.test_name. test_module must always be supplied
        and if no further refinement is made with test_class and test_name then
        all tests in the module will be run. The tests will be run in a
        subprocess with environment variables specified in `envvars`.
        If given, envvars must be a map of form:
            environment variable name (str) -> value (str)
        It is most convenient to use this method in conjunction with
        @needs_subprocess as the decorator will cause the decorated test to be
        skipped unless the `SUBPROC_TEST` environment variable is set to 1
        (this special environment variable is set by this method such that the
        specified test(s) will not be skipped in the subprocess).


        Following execution in the subprocess this method will check the test(s)
        executed without error. The timeout kwarg can be used to allow more time
        for longer running tests, it defaults to 60 seconds.
        """
        themod = self.__module__
        thecls = type(self).__name__
        parts = (test_module, test_class, test_name)
        fully_qualified_test = '.'.join((x for x in parts if x is not None))
        cmd = [sys.executable, '-m', 'numba.runtests', fully_qualified_test]
        env_copy = os.environ.copy()
        env_copy['SUBPROC_TEST'] = '1'
        try:
            env_copy['COVERAGE_PROCESS_START'] = os.environ['COVERAGE_RCFILE']
        except KeyError:
            pass
        envvars = pytypes.MappingProxyType({} if envvars is None else envvars)
        env_copy.update(envvars)
        status = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, env=env_copy, universal_newlines=True)
        streams = f'\ncaptured stdout: {status.stdout}\ncaptured stderr: {status.stderr}'
        self.assertEqual(status.returncode, 0, streams)
        no_tests_ran = 'NO TESTS RAN'
        if no_tests_ran in status.stderr:
            self.skipTest(no_tests_ran)
        else:
            self.assertIn('OK', status.stderr)

    def run_test_in_subprocess(maybefunc=None, timeout=60, envvars=None):
        """Runs the decorated test in a subprocess via invoking numba's test
        runner. kwargs timeout and envvars are passed through to
        subprocess_test_runner."""

        def wrapper(func):

            def inner(self, *args, **kwargs):
                if os.environ.get('SUBPROC_TEST', None) != '1':
                    class_name = self.__class__.__name__
                    self.subprocess_test_runner(test_module=self.__module__, test_class=class_name, test_name=func.__name__, timeout=timeout, envvars=envvars)
                else:
                    func(self)
            return inner
        if isinstance(maybefunc, pytypes.FunctionType):
            return wrapper(maybefunc)
        else:
            return wrapper

    def make_dummy_type(self):
        """Use to generate a dummy type unique to this test. Returns a python
        Dummy class and a corresponding Numba type DummyType."""
        test_id = self.id()
        DummyType = type('DummyTypeFor{}'.format(test_id), (types.Opaque,), {})
        dummy_type = DummyType('my_dummy')
        register_model(DummyType)(OpaqueModel)

        class Dummy(object):
            pass

        @typeof_impl.register(Dummy)
        def typeof_dummy(val, c):
            return dummy_type

        @unbox(DummyType)
        def unbox_dummy(typ, obj, c):
            return NativeValue(c.context.get_dummy_value())
        return (Dummy, DummyType)

    def skip_if_no_external_compiler(self):
        """
        Call this to ensure the test is skipped if no suitable external compiler
        is found. This is a method on the TestCase opposed to a stand-alone
        decorator so as to make it "lazy" via runtime evaluation opposed to
        running at test-discovery time.
        """
        from numba.pycc.platform import external_compiler_works
        if not external_compiler_works():
            self.skipTest('No suitable external compiler was found.')