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
@requires_cython
class CythonizedTraitsTestCase(unittest.TestCase, UnittestTools):

    def test_simple_default_methods(self):
        code = "\nfrom traits.api import HasTraits, Str\n\nclass Test(HasTraits):\n    name = Str\n\n    def _name_default(self):\n        return 'Joe'\n\nreturn Test()\n"
        obj = self.execute(code)
        self.assertEqual(obj.name, 'Joe')

    def test_basic_events(self):
        code = '\nfrom traits.api import HasTraits, Str\n\nclass Test(HasTraits):\n    name = Str\n\nreturn Test()\n'
        obj = self.execute(code)
        with self.assertTraitChanges(obj, 'name', count=1):
            obj.name = 'changing_name'

    def test_on_trait_static_handlers(self):
        code = '\nfrom traits.api import HasTraits, Str, Int\n\nclass Test(HasTraits):\n    name = Str\n    value = Int\n\n    def _name_changed(self):\n        self.value += 1\n\nreturn Test()\n'
        obj = self.execute(code)
        with self.assertTraitChanges(obj, 'value', count=1):
            obj.name = 'changing_name'
        self.assertEqual(obj.value, 1)

    def test_on_trait_on_trait_change_decorator(self):
        code = "\nfrom traits.api import HasTraits, Str, Int, on_trait_change\n\nclass Test(HasTraits):\n    name = Str\n    value = Int\n\n    @on_trait_change('name')\n    def _update_value(self):\n        self.value += 1\n\nreturn Test()\n"
        obj = self.execute(code)
        with self.assertTraitChanges(obj, 'value', count=1):
            obj.name = 'changing_name'
        self.assertEqual(obj.value, 1)

    def test_on_trait_properties(self):
        code = "\nfrom traits.api import HasTraits, Str, Int, Property, cached_property\n\nclass Test(HasTraits):\n    name = Str\n    name_len = Property(depends_on='name')\n\n    @cached_property\n    def _get_name_len(self):\n        return len(self.name)\n\nreturn Test()\n"
        obj = self.execute(code)
        self.assertEqual(obj.name_len, len(obj.name))
        obj.name = 'Bob'
        self.assertEqual(obj.name_len, len(obj.name))

    def test_on_trait_properties_with_standard_getter(self):
        code = '\nfrom traits.api import HasTraits, Str, Int, Property\n\nclass Test(HasTraits):\n    name = Str\n\n    def _get_name_length(self):\n        return len(self.name)\n\n    name_len = Property(_get_name_length)\n\nreturn Test()\n'
        obj = self.execute(code)
        self.assertEqual(obj.name_len, len(obj.name))
        obj.name = 'Bob'
        self.assertEqual(obj.name_len, len(obj.name))

    def test_on_trait_aliasing(self):
        code = "\nfrom traits.api import HasTraits, Str, Int, Property\n\ndef Alias(name):\n    def _get_value(self):\n        return getattr(self, name)\n    def _set_value(self, value):\n        return setattr(self, name, value)\n\n    return Property(_get_value, _set_value)\n\nclass Test(HasTraits):\n    name = Str\n\n    funky_name = Alias('name')\n\nreturn Test()\n"
        obj = self.execute(code)
        self.assertEqual(obj.funky_name, obj.name)
        obj.name = 'Bob'
        self.assertEqual(obj.funky_name, obj.name)

    def test_on_trait_aliasing_different_scope(self):
        code = "\nfrom traits.api import HasTraits, Str, Int, Property\n\ndef _get_value(self, name):\n    return getattr(self, 'name')\ndef _set_value(self, name, value):\n    return setattr(self, 'name', value)\n\n\nclass Test(HasTraits):\n    name = Str\n\n    funky_name = Property(_get_value, _set_value)\n\nreturn Test()\n"
        obj = self.execute(code)
        self.assertEqual(obj.funky_name, obj.name)
        obj.name = 'Bob'
        self.assertEqual(obj.funky_name, obj.name)

    def execute(self, code):
        """
        Helper function to execute the given code under cython.inline and
        return the result.
        """
        return cython.inline(code, force=True, language_level='3')