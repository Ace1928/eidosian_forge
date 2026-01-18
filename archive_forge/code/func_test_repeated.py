from ... import tests
from .. import rio
def test_repeated(self):
    s = rio.Stanza()
    s.add('foo', 'bar')
    s.add('foo', 'foo')
    self.assertReadStanza(s, [b'foo: bar\n', b'foo: foo\n'])