import pytest
import cirq
def test_stratified_circuit_classifier_types():
    a, b, c, d = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(cirq.Moment([cirq.X(a), cirq.Y(b), cirq.X(c) ** 0.5, cirq.X(d)]))
    gate_result = cirq.stratified_circuit(circuit, categories=[cirq.X])
    cirq.testing.assert_same_circuits(gate_result, cirq.Circuit(cirq.Moment([cirq.X(a), cirq.X(d)]), cirq.Moment([cirq.Y(b), cirq.X(c) ** 0.5])))
    gate_type_result = cirq.stratified_circuit(circuit, categories=[cirq.XPowGate])
    cirq.testing.assert_same_circuits(gate_type_result, cirq.Circuit(cirq.Moment([cirq.X(a), cirq.X(c) ** 0.5, cirq.X(d)]), cirq.Moment([cirq.Y(b)])))
    operation_result = cirq.stratified_circuit(circuit, categories=[cirq.X(a)])
    cirq.testing.assert_same_circuits(operation_result, cirq.Circuit(cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.Y(b), cirq.X(c) ** 0.5, cirq.X(d)])))
    operation_type_result = cirq.stratified_circuit(circuit, categories=[cirq.GateOperation])
    cirq.testing.assert_same_circuits(operation_type_result, cirq.Circuit(cirq.Moment([cirq.X(a), cirq.Y(b), cirq.X(c) ** 0.5, cirq.X(d)])))
    predicate_result = cirq.stratified_circuit(circuit, categories=[lambda op: op.qubits == (b,)])
    cirq.testing.assert_same_circuits(predicate_result, cirq.Circuit(cirq.Moment([cirq.Y(b)]), cirq.Moment([cirq.X(a), cirq.X(d), cirq.X(c) ** 0.5])))
    with pytest.raises(TypeError, match='Unrecognized'):
        _ = cirq.stratified_circuit(circuit, categories=['unknown'])