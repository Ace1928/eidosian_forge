from __future__ import annotations
import itertools
from copy import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING
@dataclass
class layout_details:
    """
    Layout information
    """
    panel_index: int
    panel: int
    row: int
    col: int
    scale_x: int
    scale_y: int
    axis_x: bool
    axis_y: bool
    variables: dict[str, Any]
    nrow: int
    ncol: int

    @property
    def is_left(self) -> bool:
        """
        Return True if panel is on the left
        """
        return self.col == 1

    @property
    def is_right(self) -> bool:
        """
        Return True if panel is on the right
        """
        return self.col == self.ncol

    @property
    def is_top(self) -> bool:
        """
        Return True if panel is at the top
        """
        return self.row == 1

    @property
    def is_bottom(self) -> bool:
        """
        Return True if Panel is at the bottom
        """
        return self.row == self.nrow