import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns
def test_parity_control_with_line_qubits_mapped_to():
    a, b, c = cirq.LineQubit.range(3)
    a2, b2, c2 = cirq.NamedQubit.range(3, prefix='q')
    cell = cirq.interop.quirk.cells.control_cells.ParityControlCell([a, b], [cirq.Y(c) ** 0.5])
    mapped_cell = cirq.interop.quirk.cells.control_cells.ParityControlCell([a2, b2], [cirq.Y(c2) ** 0.5])
    assert cell != mapped_cell
    assert cell.with_line_qubits_mapped_to([a2, b2, c2]) == mapped_cell