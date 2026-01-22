import inspect
import unittest
from traits.api import (
class Unbool:
    """
    Object that can't be interpreted as a bool, for testing purposes.
    """

    def __bool__(self):
        raise ZeroDivisionError()