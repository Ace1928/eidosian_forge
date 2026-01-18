from yaql.language import exceptions
import yaql.tests
def test_to_upper(self):
    self.assertEqual('QQ', self.eval('qq.toUpper()'))
    self.assertEqual(u'ПРИВЕТ', self.eval(u'Привет.toUpper()'))