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