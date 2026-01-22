from unittest import mock
from typing import Optional
import cirq
from cirq.transformers.transformer_api import LogLevel
import pytest
@cirq.transformer(add_deep_support=True)
class MockTransformerClassSupportsDeep(MockTransformerClass):
    pass