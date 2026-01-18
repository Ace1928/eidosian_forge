from yaql.language import exceptions
import yaql.tests
def test_set_union(self):
    self.assertCountEqual([4, 3, 2, 1], self.eval('set(1, 2, 3).union(set(4, 2, 3))'))