from yaql.language import exceptions
import yaql.tests
def test_to_char_array(self):
    self.assertEqual(['a', 'b', 'c'], self.eval('abc.toCharArray()'))