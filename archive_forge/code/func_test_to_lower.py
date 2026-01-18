from yaql.language import exceptions
import yaql.tests
def test_to_lower(self):
    self.assertEqual('qq', self.eval('QQ.toLower()'))
    self.assertEqual(u'привет', self.eval(u'Привет.toLower()'))