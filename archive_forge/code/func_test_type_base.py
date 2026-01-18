import unittest
from traits.api import HasTraits, Subclass, TraitError, Type
def test_type_base(self):
    model = ExampleTypeModel(_class=BaseClass)
    self.assertIsInstance(model._class(), BaseClass)