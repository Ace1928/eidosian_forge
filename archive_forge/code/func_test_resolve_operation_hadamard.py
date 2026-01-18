import cirq
import cirq_web
import pytest
def test_resolve_operation_hadamard():
    mock_qubit = cirq.NamedQubit('mock')
    operation = cirq.H(mock_qubit)
    symbol_info = cirq_web.circuits.symbols.resolve_operation(operation, cirq_web.circuits.symbols.DEFAULT_SYMBOL_RESOLVERS)
    expected_labels = ['H']
    expected_colors = ['yellow']
    assert symbol_info.labels == expected_labels
    assert symbol_info.colors == expected_colors