import pytest
import cirq
import cirq_google as cg
import cirq_google.engine.engine_validator as engine_validator
def test_validate_gate_set():
    circuit = _big_circuit(4)
    engine_validator.validate_program([circuit] * 5, [{}] * 5, 1000, cg.CIRCUIT_SERIALIZER, max_size=30000)
    with pytest.raises(RuntimeError, match='Program too long'):
        engine_validator.validate_program([circuit] * 10, [{}] * 10, 1000, cg.CIRCUIT_SERIALIZER, max_size=30000)
    with pytest.raises(RuntimeError, match='Program too long'):
        engine_validator.validate_program([circuit] * 5, [{}] * 5, 1000, cg.CIRCUIT_SERIALIZER, max_size=10000)