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
def test_unsupported_format():
    qasm = 'OPENQASM 2.1;'
    parser = QasmParser()
    with pytest.raises(QasmException, match='Unsupported.*2.1.*2.0.*supported.*'):
        parser.parse(qasm)