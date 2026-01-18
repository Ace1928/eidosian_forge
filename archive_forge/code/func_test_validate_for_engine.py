import pytest
import cirq
import cirq_google as cg
import cirq_google.engine.engine_validator as engine_validator
def test_validate_for_engine():
    circuit = _big_circuit(4)
    long_circuit = cirq.Circuit([cirq.X(cirq.GridQubit(0, 0))] * 10001)
    with pytest.raises(RuntimeError, match='Provided circuit exceeds the limit'):
        engine_validator.validate_for_engine([long_circuit], [{}], 1000)
    with pytest.raises(RuntimeError, match='the number of requested total repetitions'):
        engine_validator.validate_for_engine([circuit], [{}], 10000000)
    with pytest.raises(RuntimeError, match='the number of requested total repetitions'):
        engine_validator.validate_for_engine([circuit] * 6, [{}] * 6, 1000000)
    with pytest.raises(RuntimeError, match='the number of requested total repetitions'):
        engine_validator.validate_for_engine([circuit] * 6, [{}] * 6, [4000000, 2000000, 1, 1, 1, 1])