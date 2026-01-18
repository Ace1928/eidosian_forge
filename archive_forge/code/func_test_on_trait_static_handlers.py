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
def test_on_trait_static_handlers(self):
    code = '\nfrom traits.api import HasTraits, Str, Int\n\nclass Test(HasTraits):\n    name = Str\n    value = Int\n\n    def _name_changed(self):\n        self.value += 1\n\nreturn Test()\n'
    obj = self.execute(code)
    with self.assertTraitChanges(obj, 'value', count=1):
        obj.name = 'changing_name'
    self.assertEqual(obj.value, 1)