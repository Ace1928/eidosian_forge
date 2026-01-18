import unittest
from traits.api import AbstractViewElement, HasTraits, Int, TraitError
from traits.testing.optional_dependencies import requires_traitsui
def test_view_element_superclass(self):
    from traitsui.api import ViewElement
    self.assertIsInstance(ViewElement(), AbstractViewElement)