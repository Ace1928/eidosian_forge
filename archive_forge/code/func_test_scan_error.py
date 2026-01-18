import sys, pickle
from unittest import TestCase
import simplejson as json
from simplejson.compat import text_type, b
def test_scan_error(self):
    err = None
    for t in (text_type, b):
        try:
            json.loads(t('{"asdf": "'))
        except json.JSONDecodeError:
            err = sys.exc_info()[1]
        else:
            self.fail('Expected JSONDecodeError')
        self.assertEqual(err.lineno, 1)
        self.assertEqual(err.colno, 10)