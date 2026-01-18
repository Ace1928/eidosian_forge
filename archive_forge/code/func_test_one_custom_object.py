from io import StringIO
import yaml
from unittest import mock
from cliff.formatters import yaml_format
from cliff.tests import base
from cliff.tests import test_columns
def test_one_custom_object(self):
    sf = yaml_format.YAMLFormatter()
    c = ('a', 'b', 'toDict', 'to_dict')
    d = ('A', 'B', _toDict(spam='ham'), _to_Dict(ham='eggs'))
    expected = {'a': 'A', 'b': 'B', 'toDict': {'spam': 'ham'}, 'to_dict': {'ham': 'eggs'}}
    output = StringIO()
    args = mock.Mock()
    sf.emit_one(c, d, output, args)
    actual = yaml.safe_load(output.getvalue())
    self.assertEqual(expected, actual)