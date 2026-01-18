from collections import namedtuple
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.formatting import tostr, tabular_writer, StreamIndenter
def test_tuple_list_dict(self):
    os = StringIO()
    data = {(1,): (['a', 1], 1), ('2', 3): ({1: 'a', 2: '2'}, '2')}
    tabular_writer(os, '', data.items(), ['s', 'val'], lambda k, v: v)
    ref = u"\nKey      : s                : val\n    (1,) :         ['a', 1] :   1\n('2', 3) : {1: 'a', 2: '2'} :   2\n"
    self.assertEqual(ref.strip(), os.getvalue().strip())