import ast
import io
import sys
import traceback
import testtools
from testtools.compat import (
from testtools.matchers import (
class TestTextRepr(testtools.TestCase):
    """Ensure in extending repr, basic behaviours are not being broken"""
    ascii_examples = (('\x00', "'\\x00'", "'''\\\n\\x00'''"), ('\x08', "'\\x08'", "'''\\\n\\x08'''"), ('\t', "'\\t'", "'''\\\n\\t'''"), ('\n', "'\\n'", "'''\\\n\n'''"), ('\r', "'\\r'", "'''\\\n\\r'''"), ('"', '\'"\'', '\'\'\'\\\n"\'\'\''), ("'", '"\'"', "'''\\\n\\''''"), ('\\', "'\\\\'", "'''\\\n\\\\'''"), ('\x7f', "'\\x7f'", "'''\\\n\\x7f'''"), ('\r\n', "'\\r\\n'", "'''\\\n\\r\n'''"), ('"\'', '\'"\\\'\'', '\'\'\'\\\n"\\\'\'\'\''), ('\'"', '\'\\\'"\'', '\'\'\'\\\n\'"\'\'\''), ('\\n', "'\\\\n'", "'''\\\n\\\\n'''"), ('\\\n', "'\\\\\\n'", "'''\\\n\\\\\n'''"), ("\\' ", '"\\\\\' "', "'''\\\n\\\\' '''"), ("\\'\n", '"\\\\\'\\n"', "'''\\\n\\\\'\n'''"), ('\\\'"', '\'\\\\\\\'"\'', '\'\'\'\\\n\\\\\'"\'\'\''), ("\\'''", '"\\\\\'\'\'"', "'''\\\n\\\\\\'\\'\\''''"))
    bytes_examples = ((_b('\x80'), "'\\x80'", "'''\\\n\\x80'''"), (_b('\xa0'), "'\\xa0'", "'''\\\n\\xa0'''"), (_b('À'), "'\\xc0'", "'''\\\n\\xc0'''"), (_b('ÿ'), "'\\xff'", "'''\\\n\\xff'''"), (_b('Â§'), "'\\xc2\\xa7'", "'''\\\n\\xc2\\xa7'''"))
    unicode_examples = (('\x80', "'\\x80'", "'''\\\n\\x80'''"), ('\x9f', "'\\x9f'", "'''\\\n\\x9f'''"), ('\xa0', "'\\xa0'", "'''\\\n\\xa0'''"), ('¡', "'¡'", "'''\\\n¡'''"), ('ÿ', "'ÿ'", "'''\\\nÿ'''"), ('Ā', "'Ā'", "'''\\\nĀ'''"), ('\u2028', "'\\u2028'", "'''\\\n\\u2028'''"), ('\u2029', "'\\u2029'", "'''\\\n\\u2029'''"), ('\ud800', "'\\ud800'", "'''\\\n\\ud800'''"), ('\udfff', "'\\udfff'", "'''\\\n\\udfff'''"))
    b_prefix = repr(_b(''))[:-2]
    u_prefix = repr('')[:-2]

    def test_ascii_examples_oneline_bytes(self):
        for s, expected, _ in self.ascii_examples:
            b = _b(s)
            actual = text_repr(b, multiline=False)
            self.assertEqual(actual, self.b_prefix + expected)
            self.assertEqual(ast.literal_eval(actual), b)

    def test_ascii_examples_oneline_unicode(self):
        for s, expected, _ in self.ascii_examples:
            u = s
            actual = text_repr(u, multiline=False)
            self.assertEqual(actual, self.u_prefix + expected)
            self.assertEqual(ast.literal_eval(actual), u)

    def test_ascii_examples_multiline_bytes(self):
        for s, _, expected in self.ascii_examples:
            b = _b(s)
            actual = text_repr(b, multiline=True)
            self.assertEqual(actual, self.b_prefix + expected)
            self.assertEqual(ast.literal_eval(actual), b)

    def test_ascii_examples_multiline_unicode(self):
        for s, _, expected in self.ascii_examples:
            actual = text_repr(s, multiline=True)
            self.assertEqual(actual, self.u_prefix + expected)
            self.assertEqual(ast.literal_eval(actual), s)

    def test_ascii_examples_defaultline_bytes(self):
        for s, one, multi in self.ascii_examples:
            expected = '\n' in s and multi or one
            self.assertEqual(text_repr(_b(s)), self.b_prefix + expected)

    def test_ascii_examples_defaultline_unicode(self):
        for s, one, multi in self.ascii_examples:
            expected = '\n' in s and multi or one
            self.assertEqual(text_repr(s), self.u_prefix + expected)

    def test_bytes_examples_oneline(self):
        for b, expected, _ in self.bytes_examples:
            actual = text_repr(b, multiline=False)
            self.assertEqual(actual, self.b_prefix + expected)
            self.assertEqual(ast.literal_eval(actual), b)

    def test_bytes_examples_multiline(self):
        for b, _, expected in self.bytes_examples:
            actual = text_repr(b, multiline=True)
            self.assertEqual(actual, self.b_prefix + expected)
            self.assertEqual(ast.literal_eval(actual), b)

    def test_unicode_examples_oneline(self):
        for u, expected, _ in self.unicode_examples:
            actual = text_repr(u, multiline=False)
            self.assertEqual(actual, self.u_prefix + expected)
            self.assertEqual(ast.literal_eval(actual), u)

    def test_unicode_examples_multiline(self):
        for u, _, expected in self.unicode_examples:
            actual = text_repr(u, multiline=True)
            self.assertEqual(actual, self.u_prefix + expected)
            self.assertEqual(ast.literal_eval(actual), u)