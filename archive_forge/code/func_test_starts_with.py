from yaql.language import exceptions
import yaql.tests
def test_starts_with(self):
    self.assertTrue(self.eval('ABC.startsWith(A)'))
    self.assertTrue(self.eval('ABC.startsWith(B, A)'))
    self.assertFalse(self.eval('ABC.startsWith(C)'))
    self.assertRaises(exceptions.NoMatchingMethodException, self.eval, 'ABC.startsWith(null)')