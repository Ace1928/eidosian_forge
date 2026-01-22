import itertools
import pytest
import cirq
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
def _assert_same_diagram(actual: str, expected: str):
    assert actual == expected, f'Diagram differs from the desired diagram.\n\nActual diagram:\n{actual}\n\nDesired diagram:\n{expected}\n\nHighlighted differences:\n{cirq.testing.highlight_text_differences(actual, expected)}\n'