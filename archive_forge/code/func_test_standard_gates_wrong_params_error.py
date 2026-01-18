from typing import Callable
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing as ct
from cirq import Circuit
from cirq.circuits.qasm_output import QasmUGate
from cirq.contrib.qasm_import import QasmException
from cirq.contrib.qasm_import._parser import QasmParser
@pytest.mark.parametrize(['qasm_gate', 'num_params'], [['id', 0], ['u2', 2], ['u3', 3], ['rx', 1], ['ry', 1], ['rz', 1], ['r', 2]] + [[g[0], 0] for g in single_qubit_gates])
def test_standard_gates_wrong_params_error(qasm_gate: str, num_params: int):
    qasm = f'OPENQASM 2.0;\n     include "qelib1.inc";             \n     qreg q[2];     \n     {qasm_gate}(pi, 2*pi, 3*pi, 4*pi, 5*pi) q[1];     \n'
    parser = QasmParser()
    with pytest.raises(QasmException, match=f'.*{qasm_gate}.* takes {num_params}.*got.*5.*line 4'):
        parser.parse(qasm)
    if num_params == 0:
        return
    qasm = f'OPENQASM 2.0;\n     include "qelib1.inc";             \n     qreg q[2];     \n     {qasm_gate} q[1];     \n    '
    parser = QasmParser()
    with pytest.raises(QasmException, match=f'.*{qasm_gate}.* takes {num_params}.*got.*0.*line 4'):
        parser.parse(qasm)