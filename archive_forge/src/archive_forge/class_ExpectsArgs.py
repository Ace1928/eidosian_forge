import pytest
import cirq
class ExpectsArgs:

    def _qasm_(self, args):
        return 'text'