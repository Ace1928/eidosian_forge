import pytest
import numpy as np
from cirq.contrib.qasm_import import QasmException
from cirq.contrib.qasm_import._lexer import QasmLexer
def test_qelib_inc():
    lexer = QasmLexer()
    lexer.input('include "qelib1.inc";')
    token = lexer.token()
    assert token is not None
    assert token.type == 'QELIBINC'
    assert token.value == 'include "qelib1.inc";'