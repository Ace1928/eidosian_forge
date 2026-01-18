import unittest
from traits.api import AbstractViewElement, HasTraits, Int, TraitError
from traits.testing.optional_dependencies import requires_traitsui
def test_view_definition(self):
    from traitsui.api import View
    view = View('count')

    class Model(HasTraits):
        count = Int
        my_view = view
    view_elements = Model.class_trait_view_elements()
    self.assertEqual(view_elements.content, {'my_view': view})