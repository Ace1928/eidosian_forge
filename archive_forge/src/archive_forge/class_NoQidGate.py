import pytest
import cirq
class NoQidGate:

    def _qid_shape_(self):
        return ()