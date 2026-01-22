import decimal
import sys
import unittest
from traits.api import Either, HasTraits, Int, CInt, TraitError
from traits.testing.optional_dependencies import numpy, requires_numpy
class Truncatable(object):

    def __int__(self):
        return 42