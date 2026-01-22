import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
class AsParamNestedWrapperTestCase(BasicWrapTestCase):
    """Test that _as_parameter_ is evaluated recursively.

    The _as_parameter_ attribute can be another object which
    defines its own _as_parameter_ attribute.
    """

    def wrap(self, param):
        return AsParamWrapper(AsParamWrapper(AsParamWrapper(param)))