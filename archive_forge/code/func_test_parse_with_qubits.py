import json
import urllib
import numpy as np
import pytest
import cirq
from cirq import quirk_url_to_circuit, quirk_json_to_circuit
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
def test_parse_with_qubits():
    a = cirq.GridQubit(0, 0)
    b = cirq.GridQubit(0, 1)
    c = cirq.GridQubit(0, 2)
    assert quirk_url_to_circuit('http://algassert.com/quirk#circuit={"cols":[["H"],["•","X"]]}', qubits=cirq.GridQubit.rect(4, 4)) == cirq.Circuit(cirq.H(a), cirq.X(b).controlled_by(a))
    assert quirk_url_to_circuit('http://algassert.com/quirk#circuit={"cols":[["H"],["•",1,"X"]]}', qubits=cirq.GridQubit.rect(4, 4)) == cirq.Circuit(cirq.H(a), cirq.X(c).controlled_by(a))
    with pytest.raises(IndexError, match='qubits specified'):
        _ = quirk_url_to_circuit('http://algassert.com/quirk#circuit={"cols":[["H"],["•","X"]]}', qubits=[cirq.GridQubit(0, 0)])