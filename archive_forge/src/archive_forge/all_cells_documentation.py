from typing import Iterator
from cirq.interop.quirk.cells.arithmetic_cells import generate_all_arithmetic_cell_makers
from cirq.interop.quirk.cells.cell import CellMaker
from cirq.interop.quirk.cells.control_cells import generate_all_control_cell_makers
from cirq.interop.quirk.cells.frequency_space_cells import generate_all_frequency_space_cell_makers
from cirq.interop.quirk.cells.ignored_cells import generate_all_ignored_cell_makers
from cirq.interop.quirk.cells.input_cells import generate_all_input_cell_makers
from cirq.interop.quirk.cells.input_rotation_cells import generate_all_input_rotation_cell_makers
from cirq.interop.quirk.cells.measurement_cells import generate_all_measurement_cell_makers
from cirq.interop.quirk.cells.qubit_permutation_cells import (
from cirq.interop.quirk.cells.scalar_cells import generate_all_scalar_cell_makers
from cirq.interop.quirk.cells.single_qubit_rotation_cells import (
from cirq.interop.quirk.cells.swap_cell import generate_all_swap_cell_makers
from cirq.interop.quirk.cells.unsupported_cells import generate_all_unsupported_cell_makers
Yields a `CellMaker` for every known Quirk gate, display, etc.