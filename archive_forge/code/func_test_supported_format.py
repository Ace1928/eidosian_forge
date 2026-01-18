import pytest
import numpy as np
from cirq.contrib.qasm_import import QasmException
from cirq.contrib.qasm_import._lexer import QasmLexer
def test_supported_format():
    lexer = QasmLexer()
    lexer.input('OPENQASM 2.0;')
    token = lexer.token()
    assert token is not None
    assert token.type == 'FORMAT_SPEC'
    assert token.value == '2.0'