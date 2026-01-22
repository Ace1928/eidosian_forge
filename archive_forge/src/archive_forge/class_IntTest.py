import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class IntTest(AnyTraitTest):

    def setUp(self):
        self.obj = IntTrait()
    _default_value = 99
    _good_values = [10, -10]
    _bad_values = ['ten', b'ten', [10], {'ten': 10}, (10,), None, 1j, 10.1, -10.1, '10L', '-10L', '10.1', '-10.1', b'10L', b'-10L', b'10.1', b'-10.1', '10', '-10', b'10', b'-10']
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        _good_values.extend([np.int64(10), np.int64(-10), np.int32(10), np.int32(-10), np.int_(10), np.int_(-10)])

    def coerce(self, value):
        try:
            return int(value)
        except:
            return int(float(value))