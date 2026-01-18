from yaql.language import exceptions
import yaql.tests
def test_order_by_multilevel(self):
    self.assertEqual([[1, 0], [1, 5], [2, 2]], self.eval('$.orderBy($[0]).thenBy($[1])', data=[[2, 2], [1, 5], [1, 0]]))
    self.assertEqual([[1, 5], [1, 0], [2, 2]], self.eval('$.orderBy($[0]).thenByDescending($[1])', data=[[2, 2], [1, 5], [1, 0]]))
    self.assertEqual([[2, 2], [1, 0], [1, 5]], self.eval('$.orderByDescending($[0]).thenBy($[1])', data=[[2, 2], [1, 5], [1, 0]]))
    self.assertEqual([[2, 2], [1, 5], [1, 0]], self.eval('$.orderByDescending($[0]).thenByDescending($[1])', data=[[2, 2], [1, 5], [1, 0]]))