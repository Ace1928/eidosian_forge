import pytest
import numpy as np
from cirq.contrib.qasm_import import QasmException
from cirq.contrib.qasm_import._lexer import QasmLexer
@pytest.mark.parametrize('identifier', ['b', 'CX', 'abc', 'aXY03', 'a_valid_name_with_02_digits_and_underscores'])
def test_valid_ids(identifier: str):
    lexer = QasmLexer()
    lexer.input(identifier)
    token = lexer.token()
    assert token is not None
    assert token.type == 'ID'
    assert token.value == identifier