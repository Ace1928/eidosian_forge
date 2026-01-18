from yaql.language import exceptions
import yaql.tests
def test_substring(self):
    data = 'abcdef'
    self.assertEqual('cdef', self.eval('$.substring(2)', data=data))
    self.assertEqual('ef', self.eval('$.substring(-2)', data=data))
    self.assertEqual('cde', self.eval('$.substring(2, 3)', data=data))
    self.assertEqual('de', self.eval('$.substring(-3, 2)', data=data))
    self.assertEqual('bcdef', self.eval('$.substring(1, -1)', data=data))
    self.assertEqual('bcdef', self.eval('$.substring(-5, -1)', data=data))