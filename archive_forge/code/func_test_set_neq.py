from yaql.language import exceptions
import yaql.tests
def test_set_neq(self):
    self.assertFalse(self.eval('set(1, 2, 3) != set(3, 2, 1)'))
    self.assertTrue(self.eval('set(1, 2, 3) != set(1, 2, 3, 4)'))