from yaql.language import exceptions
import yaql.tests
def test_to_set(self):
    self.assertCountEqual([2, 1, 3], self.eval('[1, 2, 3].select($).toSet()'))
    self.assertCountEqual([2, 1, 3], self.eval('[1, 2, 3].toSet()'))