import fractions
import pytest
from cirq.testing.equals_tester import EqualsTester
class FirstClass:

    def __init__(self, val):
        self.val = val

    def __eq__(self, other):
        if not isinstance(other, FirstClass):
            return False
        return self.val == other.val