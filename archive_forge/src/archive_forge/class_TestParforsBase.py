import math
import os
import re
import dis
import numbers
import platform
import sys
import subprocess
import types as pytypes
import warnings
from functools import reduce
import numpy as np
from numpy.random import randn
import operator
from collections import defaultdict, namedtuple
import copy
from itertools import cycle, chain
import subprocess as subp
import numba.parfors.parfor
from numba import (njit, prange, parallel_chunksize,
from numba.core import (types, errors, ir, rewrites,
from numba.extending import (overload_method, register_model,
from numba.core.registry import cpu_target
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (find_callname, guard, build_definitions,
from numba.np.unsafe.ndarray import empty_inferred as unsafe_empty
from numba.core.compiler import (CompilerBase, DefaultPassBuilder)
from numba.core.compiler_machinery import register_pass, AnalysisPass
from numba.core.typed_passes import IRLegalization
from numba.tests.support import (TestCase, captured_stdout, MemoryLeakMixin,
from numba.core.extending import register_jitable
from numba.core.bytecode import _fix_LOAD_GLOBAL_arg
from numba.core import utils
import cmath
import unittest
@needs_subprocess
class TestParforsBase(TestCase):
    """
    Base class for testing parfors.
    Provides functions for compilation and three way comparison between
    python functions, njit'd functions and parfor njit'd functions.
    """
    _numba_parallel_test_ = False

    def _compile_this(self, func, sig, **flags):
        return njit(sig, **flags)(func).overloads[sig]

    def compile_parallel(self, func, sig):
        return self._compile_this(func, sig, parallel=True)

    def compile_parallel_fastmath(self, func, sig):
        return self._compile_this(func, sig, parallel=True, fastmath=True)

    def compile_njit(self, func, sig):
        return self._compile_this(func, sig)

    def compile_all(self, pyfunc, *args, **kwargs):
        sig = tuple([numba.typeof(x) for x in args])
        cpfunc = self.compile_parallel(pyfunc, sig)
        cfunc = self.compile_njit(pyfunc, sig)
        return (cfunc, cpfunc)

    def check_parfors_vs_others(self, pyfunc, cfunc, cpfunc, *args, **kwargs):
        """
        Checks python, njit and parfor impls produce the same result.

        Arguments:
            pyfunc - the python function to test
            cfunc - CompilerResult from njit of pyfunc
            cpfunc - CompilerResult from njit(parallel=True) of pyfunc
            args - arguments for the function being tested
        Keyword Arguments:
            scheduler_type - 'signed', 'unsigned' or None, default is None.
                           Supply in cases where the presence of a specific
                           scheduler is to be asserted.
            fastmath_pcres - a fastmath parallel compile result, if supplied
                             will be run to make sure the result is correct
            check_arg_equality - some functions need to check that a
                                 parameter is modified rather than a certain
                                 value returned.  If this keyword argument
                                 is supplied, it should be a list of
                                 comparison functions such that the i'th
                                 function in the list is used to compare the
                                 i'th parameter of the njit and parallel=True
                                 functions against the i'th parameter of the
                                 standard Python function, asserting if they
                                 differ.  The length of this list must be equal
                                 to the number of parameters to the function.
                                 The null comparator is available for use
                                 when you do not desire to test if some
                                 particular parameter is changed.
            Remaining kwargs are passed to np.testing.assert_almost_equal
        """
        scheduler_type = kwargs.pop('scheduler_type', None)
        check_fastmath = kwargs.pop('check_fastmath', None)
        fastmath_pcres = kwargs.pop('fastmath_pcres', None)
        check_scheduling = kwargs.pop('check_scheduling', True)
        check_args_for_equality = kwargs.pop('check_arg_equality', None)

        def copy_args(*args):
            if not args:
                return tuple()
            new_args = []
            for x in args:
                if isinstance(x, np.ndarray):
                    new_args.append(x.copy('k'))
                elif isinstance(x, np.number):
                    new_args.append(x.copy())
                elif isinstance(x, numbers.Number):
                    new_args.append(x)
                elif x is None:
                    new_args.append(x)
                elif isinstance(x, tuple):
                    new_args.append(copy.deepcopy(x))
                elif isinstance(x, list):
                    new_args.append(x[:])
                else:
                    raise ValueError('Unsupported argument type encountered')
            return tuple(new_args)
        py_args = copy_args(*args)
        py_expected = pyfunc(*py_args)
        njit_args = copy_args(*args)
        njit_output = cfunc.entry_point(*njit_args)
        parfor_args = copy_args(*args)
        parfor_output = cpfunc.entry_point(*parfor_args)
        if check_args_for_equality is None:
            np.testing.assert_almost_equal(njit_output, py_expected, **kwargs)
            np.testing.assert_almost_equal(parfor_output, py_expected, **kwargs)
            self.assertEqual(type(njit_output), type(parfor_output))
        else:
            assert len(py_args) == len(check_args_for_equality)
            for pyarg, njitarg, parforarg, argcomp in zip(py_args, njit_args, parfor_args, check_args_for_equality):
                argcomp(njitarg, pyarg, **kwargs)
                argcomp(parforarg, pyarg, **kwargs)
        if check_scheduling:
            self.check_scheduling(cpfunc, scheduler_type)
        if fastmath_pcres is not None:
            parfor_fastmath_output = fastmath_pcres.entry_point(*copy_args(*args))
            np.testing.assert_almost_equal(parfor_fastmath_output, py_expected, **kwargs)

    def check(self, pyfunc, *args, **kwargs):
        """Checks that pyfunc compiles for *args under parallel=True and njit
        and asserts that all version execute and produce the same result"""
        cfunc, cpfunc = self.compile_all(pyfunc, *args)
        self.check_parfors_vs_others(pyfunc, cfunc, cpfunc, *args, **kwargs)

    def check_variants(self, impl, arg_gen, **kwargs):
        """Run self.check(impl, ...) on array data generated from arg_gen.
        """
        for args in arg_gen():
            with self.subTest(list(map(typeof, args))):
                self.check(impl, *args, **kwargs)

    def count_parfors_variants(self, impl, arg_gen, **kwargs):
        """Run self.countParfors(impl, ...) on array types generated from
        arg_gen.
        """
        for args in arg_gen():
            with self.subTest(list(map(typeof, args))):
                argtys = tuple(map(typeof, args))
                self.assertGreaterEqual(countParfors(impl, argtys), 1)

    def check_scheduling(self, cres, scheduler_type):
        scheduler_str = '@do_scheduling'
        if scheduler_type is not None:
            if scheduler_type in ['signed', 'unsigned']:
                scheduler_str += '_' + scheduler_type
            else:
                msg = 'Unknown scheduler_type specified: %s'
                raise ValueError(msg % scheduler_type)
        self.assertIn(scheduler_str, cres.library.get_llvm_str())

    def gen_linspace(self, n, ct):
        """Make *ct* sample 1D arrays of length *n* using np.linspace().
        """

        def gen():
            yield np.linspace(0, 1, n)
            yield np.linspace(2, 1, n)
            yield np.linspace(1, 2, n)
        src = cycle(gen())
        return [next(src) for i in range(ct)]

    def gen_linspace_variants(self, ct):
        """Make 1D, 2D, 3D variants of the data in C and F orders
        """
        yield self.gen_linspace(10, ct=ct)
        arr2ds = [x.reshape((2, 3)) for x in self.gen_linspace(n=2 * 3, ct=ct)]
        yield arr2ds
        yield [np.asfortranarray(x) for x in arr2ds]
        arr3ds = [x.reshape((2, 3, 4)) for x in self.gen_linspace(n=2 * 3 * 4, ct=ct)]
        yield arr3ds
        yield [np.asfortranarray(x) for x in arr3ds]

    def _filter_mod(self, mod, magicstr, checkstr=None):
        """ helper function to filter out modules by name"""
        filt = [x for x in mod if magicstr in x.name]
        if checkstr is not None:
            for x in filt:
                assert checkstr in str(x)
        return filt

    def _get_gufunc_modules(self, cres, magicstr, checkstr=None):
        """ gets the gufunc LLVM Modules"""
        _modules = [x for x in cres.library._codegen._engine._ee._modules]
        potential_matches = self._filter_mod(_modules, magicstr, checkstr=checkstr)
        lib_asm = cres.library.get_asm_str()
        ret = []
        for mod in potential_matches:
            if mod.name in lib_asm:
                ret.append(mod)
        return ret

    def _get_gufunc_info(self, cres, fn):
        """ helper for gufunc IR/asm generation"""
        magicstr = '__numba_parfor_gufunc'
        gufunc_mods = self._get_gufunc_modules(cres, magicstr)
        x = dict()
        for mod in gufunc_mods:
            x[mod.name] = fn(mod)
        return x

    def _get_gufunc_ir(self, cres):
        """
        Returns the IR of the gufuncs used as parfor kernels
        as a dict mapping the gufunc name to its IR.

        Arguments:
         cres - a CompileResult from `njit(parallel=True, ...)`
        """
        return self._get_gufunc_info(cres, str)

    def _get_gufunc_asm(self, cres):
        """
        Returns the assembly of the gufuncs used as parfor kernels
        as a dict mapping the gufunc name to its assembly.

        Arguments:
         cres - a CompileResult from `njit(parallel=True, ...)`
        """
        tm = cres.library._codegen._tm

        def emit_asm(mod):
            return str(tm.emit_assembly(mod))
        return self._get_gufunc_info(cres, emit_asm)

    def assert_fastmath(self, pyfunc, sig):
        """
        Asserts that the fastmath flag has some effect in that suitable
        instructions are now labelled as `fast`. Whether LLVM can actually do
        anything to optimise better now the derestrictions are supplied is
        another matter!

        Arguments:
         pyfunc - a function that contains operations with parallel semantics
         sig - the type signature of pyfunc
        """
        cres = self.compile_parallel_fastmath(pyfunc, sig)
        _ir = self._get_gufunc_ir(cres)

        def _get_fast_instructions(ir):
            splitted = ir.splitlines()
            fast_inst = []
            for x in splitted:
                m = re.search('\\bfast\\b', x)
                if m is not None:
                    fast_inst.append(x)
            return fast_inst

        def _assert_fast(instrs):
            ops = ('fadd', 'fsub', 'fmul', 'fdiv', 'frem', 'fcmp', 'call')
            for inst in instrs:
                count = 0
                for op in ops:
                    match = op + ' fast'
                    if match in inst:
                        count += 1
                self.assertTrue(count > 0)
        for name, guir in _ir.items():
            inst = _get_fast_instructions(guir)
            _assert_fast(inst)