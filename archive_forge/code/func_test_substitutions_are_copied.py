from Cython.TestUtils import CythonTest
from Cython.Compiler.TreeFragment import *
from Cython.Compiler.Nodes import *
from Cython.Compiler.UtilNodes import *
def test_substitutions_are_copied(self):
    T = self.fragment(u'y + y').substitute({'y': NameNode(pos=None, name='x')})
    self.assertEqual('x', T.stats[0].expr.operand1.name)
    self.assertEqual('x', T.stats[0].expr.operand2.name)
    self.assertTrue(T.stats[0].expr.operand1 is not T.stats[0].expr.operand2)