from yaql.language import exceptions
import yaql.tests
def test_set_add(self):
    self.assertCountEqual([4, 1, 2, 3], self.eval('set(1, 2, 3).add(4)'))
    self.assertCountEqual([4, 1, 2, 3, 5], self.eval('set(1, 2, 3).add(4, 5)'))
    self.assertCountEqual([1, 3, 2, [1, 2]], self.eval('set(1, 2, 3).add([1, 2])'))
    self.assertCountEqual([4, 1, None, 2, 3, 5], self.eval('set(1, 2, 3).add(4, 5, null)'))
    self.assertTrue(self.eval('isSet(set(1, 2, 3).add(4, 5, null))'))