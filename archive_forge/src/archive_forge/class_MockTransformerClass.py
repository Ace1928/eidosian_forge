from unittest import mock
from typing import Optional
import cirq
from cirq.transformers.transformer_api import LogLevel
import pytest
@cirq.transformer()
class MockTransformerClass:

    def __init__(self):
        self.mock = mock.Mock()

    def __call__(self, circuit: cirq.AbstractCircuit, *, context: Optional[cirq.TransformerContext]=None) -> cirq.Circuit:
        self.mock(circuit, context)
        return circuit.unfreeze()