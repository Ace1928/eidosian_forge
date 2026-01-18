from yaql.language import exceptions
import yaql.tests
def test_trim_left(self):
    self.assertEqual('x  ', self.eval("'  x  '.trimLeft()"))
    self.assertEqual('xba', self.eval("'abxba'.trimLeft(ab)"))