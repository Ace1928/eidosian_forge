import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class NewInstanceTest(AnyTraitTest):

    def setUp(self):
        self.obj = NewInstanceTrait()
    _default_value = ntrait_test1
    _good_values = [ntrait_test1, NTraitTest1(), NTraitTest2(), NTraitTest3(), None]
    _bad_values = [0, 0.0, 0j, NTraitTest1, NTraitTest2, NBadTraitTest(), b'bytes', 'string', [ntrait_test1], (ntrait_test1,), {'data': ntrait_test1}]