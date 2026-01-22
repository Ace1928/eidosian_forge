import sys
from unittest import TestCase
import simplejson as json
class TestFail(TestCase):

    def test_failures(self):
        for idx, doc in enumerate(JSONDOCS):
            idx = idx + 1
            if idx in SKIPS:
                json.loads(doc)
                continue
            try:
                json.loads(doc)
            except json.JSONDecodeError:
                pass
            else:
                self.fail('Expected failure for fail%d.json: %r' % (idx, doc))

    def test_array_decoder_issue46(self):
        for doc in [u'[,]', '[,]']:
            try:
                json.loads(doc)
            except json.JSONDecodeError:
                e = sys.exc_info()[1]
                self.assertEqual(e.pos, 1)
                self.assertEqual(e.lineno, 1)
                self.assertEqual(e.colno, 2)
            except Exception:
                e = sys.exc_info()[1]
                self.fail('Unexpected exception raised %r %s' % (e, e))
            else:
                self.fail("Unexpected success parsing '[,]'")

    def test_truncated_input(self):
        test_cases = [('', 'Expecting value', 0), ('[', "Expecting value or ']'", 1), ('[42', "Expecting ',' delimiter", 3), ('[42,', 'Expecting value', 4), ('["', 'Unterminated string starting at', 1), ('["spam', 'Unterminated string starting at', 1), ('["spam"', "Expecting ',' delimiter", 7), ('["spam",', 'Expecting value', 8), ('{', "Expecting property name enclosed in double quotes or '}'", 1), ('{"', 'Unterminated string starting at', 1), ('{"spam', 'Unterminated string starting at', 1), ('{"spam"', "Expecting ':' delimiter", 7), ('{"spam":', 'Expecting value', 8), ('{"spam":42', "Expecting ',' delimiter", 10), ('{"spam":42,', 'Expecting property name enclosed in double quotes', 11), ('"', 'Unterminated string starting at', 0), ('"spam', 'Unterminated string starting at', 0), ('[,', 'Expecting value', 1), ('--', 'Expecting value', 0), ('"\x18d', 'Invalid control character %r', 1)]
        for data, msg, idx in test_cases:
            try:
                json.loads(data)
            except json.JSONDecodeError:
                e = sys.exc_info()[1]
                self.assertEqual(e.msg[:len(msg)], msg, "%r doesn't start with %r for %r" % (e.msg, msg, data))
                self.assertEqual(e.pos, idx, 'pos %r != %r for %r' % (e.pos, idx, data))
            except Exception:
                e = sys.exc_info()[1]
                self.fail('Unexpected exception raised %r %s' % (e, e))
            else:
                self.fail("Unexpected success parsing '%r'" % (data,))