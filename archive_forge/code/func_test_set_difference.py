from yaql.language import exceptions
import yaql.tests
def test_set_difference(self):
    self.assertCountEqual([4, 1], self.eval('set(1, 2, 3, 4).difference(set(2, 3))'))