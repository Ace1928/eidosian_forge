from unittest import mock
from typing import Optional
import cirq
from cirq.transformers.transformer_api import LogLevel
import pytest
@pytest.mark.parametrize('transformer', [MockTransformerClassWithDefaults(), make_transformer_func_with_defaults()])
def test_transformer_decorator_with_defaults(transformer):
    circuit = cirq.Circuit(cirq.X(cirq.NamedQubit('a')))
    context = cirq.TransformerContext(tags_to_ignore=('tags', 'to', 'ignore'))
    transformer(circuit)
    transformer.mock.assert_called_with(circuit, cirq.TransformerContext(), 0.0001, CustomArg())
    transformer(circuit, context=context, atol=0.001)
    transformer.mock.assert_called_with(circuit, context, 0.001, CustomArg())
    transformer(circuit, context=context, custom_arg=CustomArg(10))
    transformer.mock.assert_called_with(circuit, context, 0.0001, CustomArg(10))
    transformer(circuit, context=context, atol=0.01, custom_arg=CustomArg(12))
    transformer.mock.assert_called_with(circuit, context, 0.01, CustomArg(12))