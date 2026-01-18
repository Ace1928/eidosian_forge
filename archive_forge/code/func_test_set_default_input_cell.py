import pytest
from cirq import quirk_url_to_circuit
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
from cirq.interop.quirk.cells.input_cells import SetDefaultInputCell
def test_set_default_input_cell():
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":11}],["+=A4"]]}', maps={0: 11, 4: 15, 5: 0})
    assert_url_to_circuit_returns('{"cols":[["+=A4",{"id":"setA","arg":11}]]}', maps={0: 11, 4: 15, 5: 0})
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":0}],["+=A4",{"id":"setA","arg":11}]]}', maps={0: 11, 4: 15, 5: 0})
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":11}],["+=A4",{"id":"setA","arg":0}]]}', maps={0: 0, 4: 4, 5: 5})
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":1}],["+=A4"],[{"id":"setA","arg":4}],["+=A4"]]}', maps={0: 5})
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":1}],["+=A2",1,"+=A2"],["+=A2",1,"+=A2"]]}', maps={0: 10, 9: 3})
    with pytest.raises(ValueError, match='Missing input'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":[["+=A2"],[{"id":"setA","arg":1}]]}')