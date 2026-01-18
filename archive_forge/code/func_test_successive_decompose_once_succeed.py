import itertools
from typing import Optional
from unittest import mock
import pytest
import cirq
@mock.patch('cirq.protocols.decompose_protocol._CONTEXT_COUNTER', itertools.count())
def test_successive_decompose_once_succeed():
    op = G2()(cirq.NamedQubit('q'))
    d1 = cirq.decompose_once(op)
    d2 = cirq.decompose_once(d1[0])
    assert d2 == [cirq.CNOT(cirq.ops.CleanQubit(0, prefix='_decompose_protocol_0'), cirq.ops.CleanQubit(0, prefix='_decompose_protocol_1'))]