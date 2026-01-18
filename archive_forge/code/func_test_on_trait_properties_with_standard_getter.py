import unittest
from traits.testing.unittest_tools import UnittestTools
from traits.testing.optional_dependencies import cython, requires_cython
from traits.api import HasTraits, Str
from traits.api import HasTraits, Str
from traits.api import HasTraits, Str, Int
from traits.api import HasTraits, Str, Int, on_trait_change
from traits.api import HasTraits, Str, Int, Property, cached_property
from traits.api import HasTraits, Str, Int, Property
from traits.api import HasTraits, Str, Int, Property
from traits.api import HasTraits, Str, Int, Property
def test_on_trait_properties_with_standard_getter(self):
    code = '\nfrom traits.api import HasTraits, Str, Int, Property\n\nclass Test(HasTraits):\n    name = Str\n\n    def _get_name_length(self):\n        return len(self.name)\n\n    name_len = Property(_get_name_length)\n\nreturn Test()\n'
    obj = self.execute(code)
    self.assertEqual(obj.name_len, len(obj.name))
    obj.name = 'Bob'
    self.assertEqual(obj.name_len, len(obj.name))