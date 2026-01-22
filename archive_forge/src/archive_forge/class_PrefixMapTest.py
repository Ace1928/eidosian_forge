import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class PrefixMapTest(AnyTraitTest):

    def setUp(self):
        self.obj = PrefixMapTrait()
    _default_value = 'one'
    _good_values = ['o', 'on', 'one', 'tw', 'two', 'th', 'thr', 'thre', 'three']
    _mapped_values = [1, 1, 1, 2, 2, 3, 3, 3]
    _bad_values = ['t', 'one ', ' two', 1, None]

    def coerce(self, value):
        return {'o': 'one', 'on': 'one', 'tw': 'two', 'th': 'three'}[value[:2]]