from yaql.language import exceptions
import yaql.tests
def test_verbatim_strings(self):
    self.assertEqual('c:\\f\\x', self.eval('`c:\\f\\x`'))
    self.assertEqual('`', self.eval('`\\``'))
    self.assertEqual('\\n', self.eval('`\\n`'))
    self.assertEqual('\\\\', self.eval('`\\\\`'))