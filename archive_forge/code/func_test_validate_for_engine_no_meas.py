import pytest
import cirq
import cirq_google as cg
import cirq_google.engine.engine_validator as engine_validator
def test_validate_for_engine_no_meas():
    circuit = cirq.Circuit(cirq.X(cirq.GridQubit(0, 0)))
    with pytest.raises(RuntimeError, match='Code must measure at least one qubit.'):
        engine_validator.validate_for_engine([circuit] * 6, [{}] * 6, 1000)