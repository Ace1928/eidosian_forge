from collections import namedtuple
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.formatting import tostr, tabular_writer, StreamIndenter
def test_noprefix(self):
    OUT1 = StringIO()
    OUT2 = StreamIndenter(OUT1)
    OUT2.write('Hello?\nHello, world!')
    self.assertEqual('    Hello?\n    Hello, world!', OUT2.getvalue())