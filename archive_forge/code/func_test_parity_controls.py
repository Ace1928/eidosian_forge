import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
def test_parity_controls():
    a, b, c, d, e = cirq.LineQubit.range(5)
    assert_url_to_circuit_returns('{"cols":[["Y","xpar","ypar","zpar","Z"]]}', cirq.Circuit(cirq.Y(b) ** 0.5, cirq.X(c) ** (-0.5), cirq.CNOT(c, b), cirq.CNOT(d, b), cirq.Y(a).controlled_by(b), cirq.Z(e).controlled_by(b), cirq.CNOT(d, b), cirq.CNOT(c, b), cirq.X(c) ** 0.5, cirq.Y(b) ** (-0.5)))