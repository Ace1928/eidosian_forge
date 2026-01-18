import pytest
import sympy
import cirq
def test_tagged_circuits():
    q = cirq.LineQubit(0)
    ops = [cirq.X(q), cirq.H(q)]
    tags = [sympy.Symbol('a'), 'b']
    circuit = cirq.Circuit(ops)
    frozen_circuit = cirq.FrozenCircuit(ops)
    tagged_circuit = cirq.FrozenCircuit(ops, tags=tags)
    assert tagged_circuit.tags == tuple(tags)
    assert circuit == frozen_circuit != tagged_circuit
    assert cirq.approx_eq(circuit, frozen_circuit)
    assert cirq.approx_eq(frozen_circuit, tagged_circuit)
    assert hash(frozen_circuit) != hash(tagged_circuit)
    cirq.testing.assert_equivalent_repr(tagged_circuit)
    cirq.testing.assert_json_roundtrip_works(tagged_circuit)
    assert frozen_circuit.with_tags() is frozen_circuit
    assert frozen_circuit.with_tags(*tags) == tagged_circuit
    assert tagged_circuit.with_tags('c') == cirq.FrozenCircuit(ops, tags=[*tags, 'c'])
    assert tagged_circuit.untagged == frozen_circuit
    assert frozen_circuit.untagged is frozen_circuit
    assert cirq.is_parameterized(frozen_circuit) is False
    assert cirq.is_parameterized(tagged_circuit) is True
    assert cirq.parameter_names(tagged_circuit) == {'a'}
    assert str(frozen_circuit) == str(tagged_circuit)