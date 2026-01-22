import unittest
from numba import njit
from numba.core.funcdesc import PythonFunctionDescriptor, default_mangler
from numba.core.compiler import run_frontend
from numba.core.itanium_mangler import mangle_abi_tag
class TestModule(unittest.TestCase):

    def test_module_not_in_namespace(self):
        """ Test of trying to run a compiled function
        where the module from which the function is being compiled
        doesn't exist in the namespace.
        """
        filename = 'test.py'
        name = 'mypackage'
        code = '\ndef f(x):\n    return x\n'
        objs = dict(__file__=filename, __name__=name)
        compiled = compile(code, filename, 'exec')
        exec(compiled, objs)
        compiled_f = njit(objs['f'])
        self.assertEqual(compiled_f(3), 3)