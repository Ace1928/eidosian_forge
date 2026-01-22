import pytest
import cirq
class ReturnsText:

    def _qasm_(self):
        return 'text'