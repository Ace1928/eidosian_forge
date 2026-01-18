from yaql.language import exceptions
import yaql.tests
def test_set_len(self):
    self.assertEqual(3, self.eval('set(1, 2, 3).len()'))
    self.assertEqual(3, self.eval('len(set(1, 2, 3))'))