import fractions
import pytest
from cirq.testing.equals_tester import EqualsTester
class DefaultHashImplementation:
    __hash__ = object.__hash__

    def __init__(self):
        self.x = 1

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.x == other.x

    def __ne__(self, other):
        return not self == other