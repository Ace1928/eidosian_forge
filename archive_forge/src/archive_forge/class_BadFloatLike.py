import unittest
from traits.api import (
from traits.testing.optional_dependencies import numpy, requires_numpy
class BadFloatLike(object):
    """
    Object whose __float__ method raises something other than TypeError.
    """

    def __float__(self):
        raise ZeroDivisionError('bogus error')