import fractions
import pytest
from cirq.testing.equals_tester import EqualsTester
class ThirdClass:

    def __init__(self, val):
        self.val = val

    def __eq__(self, other):
        if not isinstance(other, ThirdClass):
            return NotImplemented
        return self.val == other.val