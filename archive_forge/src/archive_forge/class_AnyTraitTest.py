import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class AnyTraitTest(BaseTest, unittest.TestCase):

    def setUp(self):
        self.obj = AnyTrait()
    _default_value = None
    _good_values = [10.0, b'ten', 'ten', [10], {'ten': 10}, (10,), None, 1j]
    _mapped_values = []
    _bad_values = []