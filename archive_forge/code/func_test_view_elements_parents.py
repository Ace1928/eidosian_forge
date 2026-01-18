import unittest
from traits.api import AbstractViewElement, HasTraits, Int, TraitError
from traits.testing.optional_dependencies import requires_traitsui
def test_view_elements_parents(self):
    from traitsui.api import View

    class Model(HasTraits):
        count = Int
        my_view = View('count')

    class ModelSubclass(Model):
        total = Int
        my_view = View('count', 'total')
    view_elements = ModelSubclass.class_trait_view_elements()
    parent_view_elements = Model.class_trait_view_elements()
    self.assertEqual(view_elements.parents[0], parent_view_elements)