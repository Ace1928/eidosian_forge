import copy
import warnings
import numpy as np
import numba
from numba.core.transforms import find_setupwiths, with_lifting
from numba.core.withcontexts import bypass_context, call_context, objmode_context
from numba.core.bytecode import FunctionIdentity, ByteCode
from numba.core.interpreter import Interpreter
from numba.core import errors
from numba.core.registry import cpu_target
from numba.core.compiler import compile_ir, DEFAULT_FLAGS
from numba import njit, typeof, objmode, types
from numba.core.extending import overload
from numba.tests.support import (MemoryLeak, TestCase, captured_stdout,
from numba.core.utils import PYVERSION
from numba.experimental import jitclass
import unittest
def test_objmode_reflected_set(self):
    ret_type = typeof({1, 2, 3, 4, 5})

    @njit
    def test2():
        with objmode(result=ret_type):
            result = {1, 2, 3, 4, 5}
        return result
    with self.assertRaises(errors.CompilerError) as raises:
        test2()
    self.assertRegex(str(raises.exception), "Objmode context failed. Argument 'result' is declared as an unsupported type: reflected set\\(int(32|64)\\). Reflected types are not supported.")