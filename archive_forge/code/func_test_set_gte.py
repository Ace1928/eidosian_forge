from yaql.language import exceptions
import yaql.tests
def test_set_gte(self):
    self.assertFalse(self.eval('set(1, 2, 4) >= set(1, 2, 3)'))
    self.assertTrue(self.eval('set(1, 2, 3) >= set(1, 2, 3)'))