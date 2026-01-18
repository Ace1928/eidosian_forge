import sys, pickle
from unittest import TestCase
import simplejson as json
from simplejson.compat import text_type, b
def test_not_serializable(self):
    try:
        json.dumps(json)
    except TypeError:
        err = sys.exc_info()[1]
    else:
        self.fail('Expected TypeError')
    self.assertEqual(str(err), 'Object of type module is not JSON serializable')