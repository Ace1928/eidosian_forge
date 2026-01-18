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
def test_simple_default_methods(self):
    code = "\nfrom traits.api import HasTraits, Str\n\nclass Test(HasTraits):\n    name = Str\n\n    def _name_default(self):\n        return 'Joe'\n\nreturn Test()\n"
    obj = self.execute(code)
    self.assertEqual(obj.name, 'Joe')