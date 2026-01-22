from __future__ import annotations
import logging # isort:skip
from .enums import (
from .has_props import HasProps
from .properties import (
class ScalarTextProps(HasProps):
    """ Properties relevant to rendering text.

    Mirrors the BokehJS ``properties.Text`` class.
    """
    text_color = Nullable(Color, default='#444444', help=_color_help % 'fill text')
    text_outline_color = Nullable(Color, default=None, help=_color_help % 'outline text')
    text_alpha = Alpha(help=_alpha_help % 'fill text')
    text_font = String(default='helvetica', help=_text_font_help)
    text_font_size = FontSize('16px')
    text_font_style = Enum(FontStyle, default='normal', help=_text_font_style_help)
    text_align = Enum(TextAlign, default='left', help=_text_align_help)
    text_baseline = Enum(TextBaseline, default='bottom', help=_text_baseline_help)
    text_line_height = Float(default=1.2, help=_text_line_height_help)