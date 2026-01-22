from __future__ import annotations
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.key_binding import KeyBindingsBase
from prompt_toolkit.mouse_events import MouseEvent
from .containers import Container, ScrollOffsets
from .dimension import AnyDimension, Dimension, sum_layout_dimensions, to_dimension
from .mouse_handlers import MouseHandler, MouseHandlers
from .screen import Char, Screen, WritePosition
def _clip_point_to_visible_area(self, point: Point, write_position: WritePosition) -> Point:
    """
        Ensure that the cursor and menu positions always are always reported
        """
    if point.x < write_position.xpos:
        point = point._replace(x=write_position.xpos)
    if point.y < write_position.ypos:
        point = point._replace(y=write_position.ypos)
    if point.x >= write_position.xpos + write_position.width:
        point = point._replace(x=write_position.xpos + write_position.width - 1)
    if point.y >= write_position.ypos + write_position.height:
        point = point._replace(y=write_position.ypos + write_position.height - 1)
    return point