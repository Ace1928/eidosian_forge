import unittest
from traits.api import AbstractViewElement, HasTraits, Int, TraitError
from traits.testing.optional_dependencies import requires_traitsui
def test_trait_views(self):
    from traitsui.api import View
    view = View('count')

    class Model(HasTraits):
        count = Int
        my_view = view
    m = Model()
    views = m.trait_views()
    self.assertEqual(views, ['my_view'])