import sys
from unittest import TestCase
import simplejson as json
import simplejson.decoder
from simplejson.compat import b, PY3
def assertScan(given, expect, test_utf8=True):
    givens = [given]
    if not PY3 and test_utf8:
        givens.append(given.encode('utf8'))
    for given in givens:
        res, count = scanstring(given, 1, None, True)
        self.assertEqual(len(given), count)
        self.assertEqual(res, expect)