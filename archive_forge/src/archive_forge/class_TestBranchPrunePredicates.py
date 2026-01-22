import collections
import types as pytypes
import numpy as np
from numba.core.compiler import run_frontend, Flags, StateDict
from numba import jit, njit, literal_unroll
from numba.core import types, errors, ir, rewrites, ir_utils, utils, cpu
from numba.core import postproc
from numba.core.inline_closurecall import InlineClosureCallPass
from numba.tests.support import (TestCase, MemoryLeakMixin, SerialMixin,
from numba.core.analysis import dead_branch_prune, rewrite_semantic_constants
from numba.core.untyped_passes import (ReconstructSSA, TranslateByteCode,
from numba.core.compiler import DefaultPassBuilder, CompilerBase, PassManager
class TestBranchPrunePredicates(TestBranchPruneBase, SerialMixin):
    _TRUTHY = (1, 'String', True, 7.4, 3j)
    _FALSEY = (0, '', False, 0.0, 0j, None)

    def _literal_const_sample_generator(self, pyfunc, consts):
        """
        This takes a python function, pyfunc, and manipulates its co_const
        __code__ member to create a new function with different co_consts as
        supplied in argument consts.

        consts is a dict {index: value} of co_const tuple index to constant
        value used to update a pyfunc clone's co_const.
        """
        pyfunc_code = pyfunc.__code__
        co_consts = {k: v for k, v in enumerate(pyfunc_code.co_consts)}
        for k, v in consts.items():
            co_consts[k] = v
        new_consts = tuple([v for _, v in sorted(co_consts.items())])
        new_code = pyfunc_code.replace(co_consts=new_consts)
        return pytypes.FunctionType(new_code, globals())

    def test_literal_const_code_gen(self):

        def impl(x):
            _CONST1 = 'PLACEHOLDER1'
            if _CONST1:
                return 3.14159
            else:
                _CONST2 = 'PLACEHOLDER2'
            return _CONST2 + 4
        new = self._literal_const_sample_generator(impl, {1: 0, 3: 20})
        iconst = impl.__code__.co_consts
        nconst = new.__code__.co_consts
        self.assertEqual(iconst, (None, 'PLACEHOLDER1', 3.14159, 'PLACEHOLDER2', 4))
        self.assertEqual(nconst, (None, 0, 3.14159, 20, 4))
        self.assertEqual(impl(None), 3.14159)
        self.assertEqual(new(None), 24)

    def test_single_if_const(self):

        def impl(x):
            _CONST1 = 'PLACEHOLDER1'
            if _CONST1:
                return 3.14159
        for c_inp, prune in ((self._TRUTHY, False), (self._FALSEY, True)):
            for const in c_inp:
                func = self._literal_const_sample_generator(impl, {1: const})
                self.assert_prune(func, (types.NoneType('none'),), [prune], None)

    def test_single_if_negate_const(self):

        def impl(x):
            _CONST1 = 'PLACEHOLDER1'
            if not _CONST1:
                return 3.14159
        for c_inp, prune in ((self._TRUTHY, False), (self._FALSEY, True)):
            for const in c_inp:
                func = self._literal_const_sample_generator(impl, {1: const})
                self.assert_prune(func, (types.NoneType('none'),), [prune], None)

    def test_single_if_else_const(self):

        def impl(x):
            _CONST1 = 'PLACEHOLDER1'
            if _CONST1:
                return 3.14159
            else:
                return 1.61803
        for c_inp, prune in ((self._TRUTHY, False), (self._FALSEY, True)):
            for const in c_inp:
                func = self._literal_const_sample_generator(impl, {1: const})
                self.assert_prune(func, (types.NoneType('none'),), [prune], None)

    def test_single_if_else_negate_const(self):

        def impl(x):
            _CONST1 = 'PLACEHOLDER1'
            if not _CONST1:
                return 3.14159
            else:
                return 1.61803
        for c_inp, prune in ((self._TRUTHY, False), (self._FALSEY, True)):
            for const in c_inp:
                func = self._literal_const_sample_generator(impl, {1: const})
                self.assert_prune(func, (types.NoneType('none'),), [prune], None)

    def test_single_if_freevar(self):
        for c_inp, prune in ((self._TRUTHY, False), (self._FALSEY, True)):
            for const in c_inp:

                def func(x):
                    if const:
                        return (3.14159, const)
                self.assert_prune(func, (types.NoneType('none'),), [prune], None)

    def test_single_if_negate_freevar(self):
        for c_inp, prune in ((self._TRUTHY, False), (self._FALSEY, True)):
            for const in c_inp:

                def func(x):
                    if not const:
                        return (3.14159, const)
                self.assert_prune(func, (types.NoneType('none'),), [prune], None)

    def test_single_if_else_freevar(self):
        for c_inp, prune in ((self._TRUTHY, False), (self._FALSEY, True)):
            for const in c_inp:

                def func(x):
                    if const:
                        return (3.14159, const)
                    else:
                        return (1.61803, const)
                self.assert_prune(func, (types.NoneType('none'),), [prune], None)

    def test_single_if_else_negate_freevar(self):
        for c_inp, prune in ((self._TRUTHY, False), (self._FALSEY, True)):
            for const in c_inp:

                def func(x):
                    if not const:
                        return (3.14159, const)
                    else:
                        return (1.61803, const)
                self.assert_prune(func, (types.NoneType('none'),), [prune], None)

    def test_single_if_global(self):
        global c_test_single_if_global
        for c_inp, prune in ((self._TRUTHY, False), (self._FALSEY, True)):
            for c in c_inp:
                c_test_single_if_global = c

                def func(x):
                    if c_test_single_if_global:
                        return (3.14159, c_test_single_if_global)
                self.assert_prune(func, (types.NoneType('none'),), [prune], None)

    def test_single_if_negate_global(self):
        global c_test_single_if_negate_global
        for c_inp, prune in ((self._TRUTHY, False), (self._FALSEY, True)):
            for c in c_inp:
                c_test_single_if_negate_global = c

                def func(x):
                    if c_test_single_if_negate_global:
                        return (3.14159, c_test_single_if_negate_global)
                self.assert_prune(func, (types.NoneType('none'),), [prune], None)

    def test_single_if_else_global(self):
        global c_test_single_if_else_global
        for c_inp, prune in ((self._TRUTHY, False), (self._FALSEY, True)):
            for c in c_inp:
                c_test_single_if_else_global = c

                def func(x):
                    if c_test_single_if_else_global:
                        return (3.14159, c_test_single_if_else_global)
                    else:
                        return (1.61803, c_test_single_if_else_global)
                self.assert_prune(func, (types.NoneType('none'),), [prune], None)

    def test_single_if_else_negate_global(self):
        global c_test_single_if_else_negate_global
        for c_inp, prune in ((self._TRUTHY, False), (self._FALSEY, True)):
            for c in c_inp:
                c_test_single_if_else_negate_global = c

                def func(x):
                    if not c_test_single_if_else_negate_global:
                        return (3.14159, c_test_single_if_else_negate_global)
                    else:
                        return (1.61803, c_test_single_if_else_negate_global)
                self.assert_prune(func, (types.NoneType('none'),), [prune], None)

    def test_issue_5618(self):

        @njit
        def foo():
            values = np.zeros(1)
            tmp = 666
            if tmp:
                values[0] = tmp
            return values
        self.assertPreciseEqual(foo.py_func()[0], 666.0)
        self.assertPreciseEqual(foo()[0], 666.0)