from yaql.language import exceptions
import yaql.tests
def test_split_at(self):
    self.assertEqual([[1, 2], [3, 4, 5]], self.eval('range(1, 6).splitAt(2)'))