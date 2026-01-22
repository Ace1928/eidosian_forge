import itertools
import pytest
import cirq
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
def _curve_pieces_diagram(chars: BoxDrawCharacterSet) -> BlockDiagramDrawer:
    d = BlockDiagramDrawer()
    for x in range(4):
        for y in range(4):
            block = d.mutable_block(x * 2, y * 2)
            block.horizontal_alignment = 0.5
            block.draw_curve(chars, top=bool(y & 1), bottom=bool(y & 2), left=bool(x & 2), right=bool(x & 1))
    return d