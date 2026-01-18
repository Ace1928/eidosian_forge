from yaql.language import exceptions
import yaql.tests
def test_set_subtraction(self):
    self.assertCountEqual([4, 1], self.eval('set(1, 2, 3, 4) - set(2, 3)'))
    self.assertTrue(self.eval('isSet(set(1, 2, 3, 4) - set(2, 3))'))