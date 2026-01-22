from ... import tests
from .. import rio
class TestReadUTF8Stanza(tests.TestCase):
    module = None

    def assertReadStanza(self, result, line_iter):
        s = self.module._read_stanza_utf8(line_iter)
        self.assertEqual(result, s)
        if s is not None:
            for tag, value in s.iter_pairs():
                self.assertIsInstance(tag, str)
                self.assertIsInstance(value, str)

    def assertReadStanzaRaises(self, exception, line_iter):
        self.assertRaises(exception, self.module._read_stanza_utf8, line_iter)

    def test_no_string(self):
        self.assertReadStanzaRaises(TypeError, [21323])

    def test_empty(self):
        self.assertReadStanza(None, [])

    def test_none(self):
        self.assertReadStanza(None, [b''])

    def test_simple(self):
        self.assertReadStanza(rio.Stanza(foo='bar'), [b'foo: bar\n', b''])

    def test_multi_line(self):
        self.assertReadStanza(rio.Stanza(foo='bar\nbla'), [b'foo: bar\n', b'\tbla\n'])

    def test_repeated(self):
        s = rio.Stanza()
        s.add('foo', 'bar')
        s.add('foo', 'foo')
        self.assertReadStanza(s, [b'foo: bar\n', b'foo: foo\n'])

    def test_invalid_early_colon(self):
        self.assertReadStanzaRaises(ValueError, [b'f:oo: bar\n'])

    def test_invalid_tag(self):
        self.assertReadStanzaRaises(ValueError, [b'f%oo: bar\n'])

    def test_continuation_too_early(self):
        self.assertReadStanzaRaises(ValueError, [b'\tbar\n'])

    def test_large(self):
        value = b'bla' * 9000
        self.assertReadStanza(rio.Stanza(foo=value.decode()), [b'foo: %s\n' % value])

    def test_non_ascii_char(self):
        self.assertReadStanza(rio.Stanza(foo='nåme'), ['foo: nåme\n'.encode()])