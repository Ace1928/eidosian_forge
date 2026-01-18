import pytest
import cirq
def test_qasm_output_args_validate():
    args = cirq.QasmArgs(version='2.0')
    args.validate_version('2.0')
    with pytest.raises(ValueError):
        args.validate_version('2.1')