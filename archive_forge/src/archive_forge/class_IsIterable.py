import pytest
import cirq
class IsIterable:

    def __iter__(self):
        yield 1
        yield 2