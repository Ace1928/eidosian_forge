import pytest
import cirq
class ReturnsExponent:

    def __pow__(self, exponent) -> int:
        return exponent