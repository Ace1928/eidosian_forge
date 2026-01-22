import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class OddIntegerTest(AnyTraitTest):

    def setUp(self):
        self.obj = OddIntegerTrait()
    _default_value = 99
    _good_values = [1, 3, 5, 7, 9, 999999999, 1.0, 3.0, 5.0, 7.0, 9.0, 999999999.0, -1, -3, -5, -7, -9, -999999999, -1.0, -3.0, -5.0, -7.0, -9.0, -999999999.0]
    _bad_values = [0, 2, -2, 1j, None, '1', [1], (1,), {1: 1}]