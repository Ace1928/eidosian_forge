import pytest
import cirq
def test_routed_circuit_with_mapping_simple():
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.Moment(cirq.SWAP(q[0], q[1]).with_tags(cirq.RoutingSwapTag()))])
    expected_diagram = '\n0: ───q(0)───×[cirq.RoutingSwapTag()]───q(1)───\n      │      │                          │\n1: ───q(1)───×──────────────────────────q(0)───'
    cirq.testing.assert_has_diagram(cirq.routed_circuit_with_mapping(circuit), expected_diagram)
    expected_diagram_with_initial_mapping = '\n0: ───a───×[cirq.RoutingSwapTag()]───b───\n      │   │                          │\n1: ───b───×──────────────────────────a───'
    cirq.testing.assert_has_diagram(cirq.routed_circuit_with_mapping(circuit, {cirq.NamedQubit('a'): q[0], cirq.NamedQubit('b'): q[1]}), expected_diagram_with_initial_mapping)
    circuit = cirq.Circuit([cirq.Moment(cirq.SWAP(q[0], q[1]))])
    expected_diagram = '\n0: ───q(0)───×───\n      │      │\n1: ───q(1)───×───'
    cirq.testing.assert_has_diagram(cirq.routed_circuit_with_mapping(circuit), expected_diagram)
    circuit = cirq.Circuit([cirq.Moment(cirq.X(q[0]).with_tags(cirq.RoutingSwapTag())), cirq.Moment(cirq.SWAP(q[0], q[1]))])
    with pytest.raises(ValueError, match='Invalid circuit. A non-SWAP gate cannot be tagged a RoutingSwapTag.'):
        cirq.routed_circuit_with_mapping(circuit)