import unittest
from traits.api import AbstractViewElement, HasTraits, Int, TraitError
from traits.testing.optional_dependencies import requires_traitsui
class ModelSubclass(Model):
    total = Int
    my_view = View('count', 'total')