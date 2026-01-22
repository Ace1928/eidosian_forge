import fractions
import pytest
from cirq.testing.equals_tester import EqualsTester
class SecondClass:

    def __init__(self, val):
        self.val = val

    def __eq__(self, other):
        if isinstance(other, (FirstClass, SecondClass)):
            return self.val == other.val
        return NotImplemented