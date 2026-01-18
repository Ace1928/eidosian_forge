from unittest import mock
from typing import Optional
import cirq
from cirq.transformers.transformer_api import LogLevel
import pytest
@pytest.mark.parametrize('context', [cirq.TransformerContext(), cirq.TransformerContext(logger=mock.Mock(), tags_to_ignore=('tag',))])
@pytest.mark.parametrize('transformer', [MockTransformerClass(), make_transformer_func()])
def test_transformer_decorator(context, transformer):
    circuit = cirq.Circuit(cirq.X(cirq.NamedQubit('a')))
    transformer(circuit, context=context)
    transformer.mock.assert_called_with(circuit, context)
    if not isinstance(context.logger, cirq.TransformerLogger):
        transformer_name = transformer.__name__ if hasattr(transformer, '__name__') else type(transformer).__name__
        context.logger.register_initial.assert_called_with(circuit, transformer_name)
        context.logger.register_final.assert_called_with(circuit, transformer_name)