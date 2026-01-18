from yaql.language import exceptions
import yaql.tests
def test_slice_where(self):
    self.assertEqual([['a', 'a'], ['b'], ['a', 'a']], self.eval('[a,a,b,a,a].sliceWhere($ != a)'))