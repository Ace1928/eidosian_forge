from yaql.language import exceptions
import yaql.tests
def test_tuples_func(self):
    self.assertEqual((1, 2), self.eval('tuple(1, 2)'))
    self.assertEqual((None,), self.eval('tuple(null)'))
    self.assertEqual((), self.eval('tuple()'))