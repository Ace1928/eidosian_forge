from yaql.language import exceptions
import yaql.tests
def test_replace_many(self):
    self.assertEqual([7, 8, 2, 3, 4], self.eval('[1, 2, 3, 4].replaceMany(0, [7, 8])'))
    self.assertEqual([7, 8, 3, 4], self.eval('[1, 2, 3, 4].replaceMany(0, [7, 8], 2)'))
    self.assertEqual([1, 7, 8], self.eval('[1, 2, 3, 4].replaceMany(1, [7, 8], -1)'))