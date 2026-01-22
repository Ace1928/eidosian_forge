from __future__ import annotations
import itertools
from copy import copy
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING
@dataclass
class strip_draw_info:
    """
    Information required to draw strips
    """
    x: float
    y: float
    ha: str
    va: str
    box_width: float
    box_height: float
    strip_text_margin: float
    strip_align: float
    position: StripPosition
    label: str
    ax: Axes
    rotation: float
    layout: layout_details