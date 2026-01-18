from yaql.language import exceptions
import yaql.tests
def test_select_many(self):
    self.assertEqual([0, 0, 1, 0, 1, 2], self.eval('range(4).selectMany(range($))'))