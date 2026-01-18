import re
from yaql.language import exceptions
from yaql import tests
from yaql import yaqlization
def test_property_access(self):
    obj = self._get_sample_class()()
    yaqlization.yaqlize(obj)
    self.assertEqual(123, self.eval('$.attr', obj))
    self.assertEqual(123, self.eval('$.prop', obj))
    self.assertEqual(123, self.eval('$?.prop', obj))
    self.assertIsNone(self.eval('$?.prop', None))
    self.assertRaises(AttributeError, self.eval, '$.invalid', obj)