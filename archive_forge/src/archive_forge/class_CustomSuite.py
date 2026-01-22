import testtools
import random
import testresources
from testresources import split_by_resources
from testresources.tests import ResultWithResourceExtensions
import unittest
class CustomSuite(unittest.TestSuite):
    """Custom TestSuite that's comparable using == and !=."""

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self._tests == other._tests

    def __ne__(self, other):
        return not self.__eq__(other)