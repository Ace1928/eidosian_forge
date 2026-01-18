from yaql.language import exceptions
import yaql.tests
def test_split_where(self):
    self.assertEqual([[], [2, 3], [5]], self.eval('range(1, 6).splitWhere($ mod 3 = 1)'))