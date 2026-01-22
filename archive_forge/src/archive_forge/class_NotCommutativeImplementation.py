import fractions
import pytest
from cirq.testing.equals_tester import EqualsTester
class NotCommutativeImplementation:

    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.x <= other.x

    def __ne__(self, other):
        return not self == other