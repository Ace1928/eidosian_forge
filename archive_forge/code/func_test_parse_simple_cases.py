import json
import urllib
import numpy as np
import pytest
import cirq
from cirq import quirk_url_to_circuit, quirk_json_to_circuit
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
def test_parse_simple_cases():
    a, b = cirq.LineQubit.range(2)
    assert quirk_url_to_circuit('http://algassert.com/quirk') == cirq.Circuit()
    assert quirk_url_to_circuit('https://algassert.com/quirk') == cirq.Circuit()
    assert quirk_url_to_circuit('https://algassert.com/quirk#') == cirq.Circuit()
    assert quirk_url_to_circuit('http://algassert.com/quirk#circuit={"cols":[]}') == cirq.Circuit()
    assert quirk_url_to_circuit('https://algassert.com/quirk#circuit={%22cols%22:[[%22H%22],[%22%E2%80%A2%22,%22X%22]]}') == cirq.Circuit(cirq.H(a), cirq.X(b).controlled_by(a))