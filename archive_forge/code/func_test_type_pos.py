from Cython.TestUtils import CythonTest
import Cython.Compiler.Errors as Errors
from Cython.Compiler.Nodes import *
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.Buffer import *
def test_type_pos(self):
    self.parse(u'cdef object[short unsigned int, 3] x')