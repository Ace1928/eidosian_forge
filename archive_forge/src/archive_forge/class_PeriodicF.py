import pytest
import cirq
@cirq.value_equality(approximate=True)
class PeriodicF:

    def __init__(self, x, n):
        self.x = x
        self.n = n

    def _value_equality_values_(self):
        return self.x

    def _value_equality_approximate_values_(self):
        return self.x % self.n