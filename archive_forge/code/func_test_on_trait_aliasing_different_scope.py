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
def test_on_trait_aliasing_different_scope(self):
    code = "\nfrom traits.api import HasTraits, Str, Int, Property\n\ndef _get_value(self, name):\n    return getattr(self, 'name')\ndef _set_value(self, name, value):\n    return setattr(self, 'name', value)\n\n\nclass Test(HasTraits):\n    name = Str\n\n    funky_name = Property(_get_value, _set_value)\n\nreturn Test()\n"
    obj = self.execute(code)
    self.assertEqual(obj.funky_name, obj.name)
    obj.name = 'Bob'
    self.assertEqual(obj.funky_name, obj.name)