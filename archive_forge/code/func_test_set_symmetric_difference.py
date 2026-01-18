from yaql.language import exceptions
import yaql.tests
def test_set_symmetric_difference(self):
    self.assertCountEqual([4, 1, 5], self.eval('set(1, 2, 3, 4).symmetricDifference(set(2, 3, 5))'))