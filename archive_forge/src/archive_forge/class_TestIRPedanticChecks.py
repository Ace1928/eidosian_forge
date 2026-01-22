import unittest
from unittest.case import TestCase
import warnings
import numpy as np
from numba import objmode
from numba.core import ir, compiler
from numba.core import errors
from numba.core.compiler import (
from numba.core.compiler_machinery import (
from numba.core.untyped_passes import (
from numba import njit
class TestIRPedanticChecks(TestCase):

    def test_var_in_scope_assumption(self):

        @register_pass(mutates_CFG=False, analysis_only=False)
        class RemoveVarInScope(FunctionPass):
            _name = '_remove_var_in_scope'

            def __init__(self):
                FunctionPass.__init__(self)

            def run_pass(self, state):
                func_ir = state.func_ir
                for blk in func_ir.blocks.values():
                    oldscope = blk.scope
                    blk.scope = ir.Scope(parent=oldscope.parent, loc=oldscope.loc)
                return True

        @register_pass(mutates_CFG=False, analysis_only=False)
        class FailPass(FunctionPass):
            _name = '_fail'

            def __init__(self, *args, **kwargs):
                FunctionPass.__init__(self)

            def run_pass(self, state):
                raise AssertionError('unreachable')

        class MyCompiler(CompilerBase):

            def define_pipelines(self):
                pm = PassManager('testing pm')
                pm.add_pass(TranslateByteCode, 'analyzing bytecode')
                pm.add_pass(IRProcessing, 'processing IR')
                pm.add_pass(RemoveVarInScope, '_remove_var_in_scope')
                pm.add_pass(ReconstructSSA, 'ssa')
                pm.add_pass(FailPass, '_fail')
                pm.finalize()
                return [pm]

        @njit(pipeline_class=MyCompiler)
        def dummy(x):
            a = 1
            b = 2
            if a < b:
                a = 2
            else:
                b = 3
            return (a, b)
        with warnings.catch_warnings():
            warnings.simplefilter('error', errors.NumbaPedanticWarning)
            with self.assertRaises(errors.NumbaIRAssumptionWarning) as raises:
                dummy(1)
            self.assertRegex(str(raises.exception), "variable '[a-z]' is not in scope")