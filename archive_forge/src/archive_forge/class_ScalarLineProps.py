from __future__ import annotations
import logging # isort:skip
from .enums import (
from .has_props import HasProps
from .properties import (
class ScalarLineProps(HasProps):
    """ Properties relevant to rendering path operations.

    Mirrors the BokehJS ``properties.Line`` class.

    """
    line_color = Nullable(Color, default='black', help=_color_help % 'stroke paths')
    line_alpha = Alpha(help=_alpha_help % 'stroke paths')
    line_width = Float(default=1, help=_line_width_help)
    line_join = Enum(LineJoin, default='bevel', help=_line_join_help)
    line_cap = Enum(LineCap, default='butt', help=_line_cap_help)
    line_dash = DashPattern(default=[], help='How should the line be dashed.')
    line_dash_offset = Int(default=0, help='The distance into the ``line_dash`` (in pixels) that the pattern should start from.')