from Cython.TestUtils import CythonTest
import Cython.Compiler.Errors as Errors
from Cython.Compiler.Nodes import *
from Cython.Compiler.ParseTreeTransforms import *
from Cython.Compiler.Buffer import *
def test_pos_after_key(self):
    self.not_parseable('Non-keyword arg following keyword arg', u'cdef object[foo=1, 2] x')