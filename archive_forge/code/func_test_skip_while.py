from yaql.language import exceptions
import yaql.tests
def test_skip_while(self):
    self.assertEqual([4, 5], self.eval('[1, 2, 3, 4, 5].skipWhile($ < 4)'))