import sys
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.deprecation import (
from pyomo.common.log import LoggingIntercept
from io import StringIO
import logging
def test_renamed_errors(self):

    class NewClass(object):
        pass
    with self.assertRaisesRegex(TypeError, "Declaring class 'DeprecatedClass' using the RenamedClass metaclass, but without specifying the __renamed__new_class__ class attribute"):

        class DeprecatedClass(metaclass=RenamedClass):
            __renamed_new_class__ = NewClass
    with self.assertRaisesRegex(DeveloperError, "Declaring class 'DeprecatedClass' using the RenamedClass metaclass, but without specifying the __renamed__version__ class attribute", normalize_whitespace=True):

        class DeprecatedClass(metaclass=RenamedClass):
            __renamed__new_class__ = NewClass