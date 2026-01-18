import unittest
from traits.api import HasTraits, Subclass, TraitError, Type
def test_subclass_derived(self):
    model = ExampleSubclassModel(_class=DerivedClass)
    self.assertIsInstance(model._class(), DerivedClass)