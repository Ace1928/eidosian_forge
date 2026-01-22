import unittest
from traits.api import (
from traits.testing.optional_dependencies import numpy, requires_numpy
class InheritsFromFloat(float):
    """
    Object that's float-like by virtue of inheriting from float.
    """
    pass