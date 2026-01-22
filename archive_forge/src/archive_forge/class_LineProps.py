from __future__ import annotations
import logging # isort:skip
from .enums import (
from .has_props import HasProps
from .properties import (
class LineProps(HasProps):
    """ Properties relevant to rendering path operations.

    Mirrors the BokehJS ``properties.LineVector`` class.

    """
    line_color = ColorSpec(default='black', help=_color_help % 'stroke paths')
    line_alpha = AlphaSpec(help=_alpha_help % 'stroke paths')
    line_width = NumberSpec(default=1, accept_datetime=False, accept_timedelta=False, help=_line_width_help)
    line_join = LineJoinSpec(default='bevel', help=_line_join_help)
    line_cap = LineCapSpec(default='butt', help=_line_cap_help)
    line_dash = DashPatternSpec(default=[], help='How should the line be dashed.')
    line_dash_offset = IntSpec(default=0, help='The distance into the ``line_dash`` (in pixels) that the pattern should start from.')