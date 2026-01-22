from typing import (
import numpy as np
from cirq import value
from cirq.circuits._block_diagram_drawer import BlockDiagramDrawer
from cirq.circuits._box_drawing_character_data import (
def _transform_coordinates(self, func: Callable[[Union[int, float], Union[int, float]], Tuple[Union[int, float], Union[int, float]]]) -> None:
    """Helper method to transformer either row or column coordinates."""

    def func_x(x: Union[int, float]) -> Union[int, float]:
        return func(x, 0)[0]

    def func_y(y: Union[int, float]) -> Union[int, float]:
        return func(0, y)[1]
    self.entries = {cast(Tuple[int, int], func(int(x), int(y))): v for (x, y), v in self.entries.items()}
    self.vertical_lines = [_VerticalLine(func_x(x), func_y(y1), func_y(y2), emph, doubled) for x, y1, y2, emph, doubled in self.vertical_lines]
    self.horizontal_lines = [_HorizontalLine(func_y(y), func_x(x1), func_x(x2), emph, doubled) for y, x1, x2, emph, doubled in self.horizontal_lines]
    self.horizontal_padding = {int(func_x(int(x))): padding for x, padding in self.horizontal_padding.items()}
    self.vertical_padding = {int(func_y(int(y))): padding for y, padding in self.vertical_padding.items()}