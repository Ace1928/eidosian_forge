import re
from yaql.language import exceptions
from yaql import tests
from yaql import yaqlization
def test_property_access_blacklist(self):
    obj = self._get_sample_class()()
    yaqlization.yaqlize(obj, blacklist=['prop'])
    self.assertEqual(123, self.eval('$.attr', obj))
    self.assertRaises(AttributeError, self.eval, '$.prop', obj)