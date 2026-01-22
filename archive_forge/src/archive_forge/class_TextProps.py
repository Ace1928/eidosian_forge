from __future__ import annotations
import logging # isort:skip
from .enums import (
from .has_props import HasProps
from .properties import (
class TextProps(HasProps):
    """ Properties relevant to rendering text.

    Mirrors the BokehJS ``properties.TextVector`` class.
    """
    text_color = ColorSpec(default='#444444', help=_color_help % 'fill text')
    text_outline_color = ColorSpec(default=None, help=_color_help % 'outline text')
    text_alpha = AlphaSpec(help=_alpha_help % 'fill text')
    text_font = StringSpec(default=value('helvetica'), help=_text_font_help)
    text_font_size = FontSizeSpec(default=value('16px'))
    text_font_style = FontStyleSpec(default='normal', help=_text_font_style_help)
    text_align = TextAlignSpec(default='left', help=_text_align_help)
    text_baseline = TextBaselineSpec(default='bottom', help=_text_baseline_help)
    text_line_height = NumberSpec(default=1.2, help=_text_line_height_help)