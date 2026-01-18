import sys
from unittest import TestCase
import simplejson as json
import simplejson.decoder
from simplejson.compat import b, PY3
def test_surrogates(self):
    scanstring = json.decoder.scanstring

    def assertScan(given, expect, test_utf8=True):
        givens = [given]
        if not PY3 and test_utf8:
            givens.append(given.encode('utf8'))
        for given in givens:
            res, count = scanstring(given, 1, None, True)
            self.assertEqual(len(given), count)
            self.assertEqual(res, expect)
    assertScan(u'"z\\ud834\\u0079x"', u'z\ud834yx')
    assertScan(u'"z\\ud834\\udd20x"', u'z𝄠x')
    assertScan(u'"z\\ud834\\ud834\\udd20x"', u'z\ud834𝄠x')
    assertScan(u'"z\\ud834x"', u'z\ud834x')
    assertScan(u'"z\\udd20x"', u'z\udd20x')
    assertScan(u'"z\ud834x"', u'z\ud834x')
    assertScan(u'"z\\ud834\udd20x12345"', u''.join([u'z\ud834', u'\udd20x12345']))
    assertScan(u'"z\ud834\\udd20x"', u''.join([u'z\ud834', u'\udd20x']))
    assertScan(u''.join([u'"z\ud834', u'\udd20x"']), u''.join([u'z\ud834', u'\udd20x']), test_utf8=False)
    self.assertRaises(ValueError, scanstring, u'"z\\ud83x"', 1, None, True)
    self.assertRaises(ValueError, scanstring, u'"z\\ud834\\udd2x"', 1, None, True)