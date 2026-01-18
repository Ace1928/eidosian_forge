from unittest import mock
from typing import Optional
import cirq
from cirq.transformers.transformer_api import LogLevel
import pytest
@pytest.mark.parametrize('transformer, supports_deep', [(MockTransformerClass(), False), (make_transformer_func(), False), (MockTransformerClassSupportsDeep(), True), (make_transformer_func(add_deep_support=True), True)])
def test_transformer_decorator_adds_support_for_deep(transformer, supports_deep):
    q = cirq.NamedQubit('q')
    c_nested_x = cirq.FrozenCircuit(cirq.X(q))
    c_nested_y = cirq.FrozenCircuit(cirq.Y(q))
    c_nested_xy = cirq.FrozenCircuit(cirq.CircuitOperation(c_nested_x).repeat(5).with_tags('ignore'), cirq.CircuitOperation(c_nested_y).repeat(7).with_tags('preserve_tag'))
    c_nested_yx = cirq.FrozenCircuit(cirq.CircuitOperation(c_nested_y).repeat(7).with_tags('ignore'), cirq.CircuitOperation(c_nested_x).repeat(5).with_tags('preserve_tag'))
    c_orig = cirq.Circuit(cirq.CircuitOperation(c_nested_xy).repeat(4), cirq.CircuitOperation(c_nested_x).repeat(5).with_tags('ignore'), cirq.CircuitOperation(c_nested_y).repeat(6), cirq.CircuitOperation(c_nested_yx).repeat(7))
    context = cirq.TransformerContext(tags_to_ignore=['ignore'], deep=True)
    transformer(c_orig, context=context)
    expected_calls = [mock.call(c_orig, context)]
    if supports_deep:
        expected_calls = [mock.call(c_nested_y, context), mock.call(c_nested_xy, context), mock.call(c_nested_y, context), mock.call(c_nested_x, context), mock.call(c_nested_yx, context), mock.call(c_orig, context)]
    transformer.mock.assert_has_calls(expected_calls)