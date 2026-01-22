import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class OldInstanceTest(AnyTraitTest):

    def setUp(self):
        self.obj = OldInstanceTrait()
    _default_value = otrait_test1
    _good_values = [otrait_test1, OTraitTest1(), OTraitTest2(), OTraitTest3(), None]
    _bad_values = [0, 0.0, 0j, OTraitTest1, OTraitTest2, OBadTraitTest(), b'bytes', 'string', [otrait_test1], (otrait_test1,), {'data': otrait_test1}]