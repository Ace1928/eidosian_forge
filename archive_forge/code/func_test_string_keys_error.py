import sys, pickle
from unittest import TestCase
import simplejson as json
from simplejson.compat import text_type, b
def test_string_keys_error(self):
    data = [{'a': 'A', 'b': (2, 4), 'c': 3.0, ('d',): 'D tuple'}]
    try:
        json.dumps(data)
    except TypeError:
        err = sys.exc_info()[1]
    else:
        self.fail('Expected TypeError')
    self.assertEqual(str(err), 'keys must be str, int, float, bool or None, not tuple')