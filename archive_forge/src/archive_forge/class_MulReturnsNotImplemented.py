import pytest
import sympy
import cirq
class MulReturnsNotImplemented:

    def __mul__(self, other):
        return NotImplemented