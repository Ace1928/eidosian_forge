from Cython.TestUtils import CythonTest
from Cython.Compiler.TreeFragment import *
from Cython.Compiler.Nodes import *
from Cython.Compiler.UtilNodes import *
def test_temps(self):
    TemplateTransform.temp_name_counter = 0
    F = self.fragment(u'\n            TMP\n            x = TMP\n        ')
    T = F.substitute(temps=[u'TMP'])
    s = T.body.stats
    self.assertTrue(isinstance(s[0].expr, TempRefNode))
    self.assertTrue(isinstance(s[1].rhs, TempRefNode))
    self.assertTrue(s[0].expr.handle is s[1].rhs.handle)