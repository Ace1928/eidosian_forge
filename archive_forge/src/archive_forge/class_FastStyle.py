from __future__ import annotations
import pathlib
import param
from bokeh.themes import Theme as _BkTheme
from ..config import config
from ..io.resources import CDN_DIST
from ..layout import Accordion
from ..reactive import ReactiveHTML
from ..viewable import Viewable
from ..widgets import Tabulator
from ..widgets.indicators import Dial, Number, String
from .base import (
class FastStyle(param.Parameterized):
    """
    The FastStyle class provides the different colors and icons used
    to style the Fast Templates.
    """
    background_color = param.String(default='#ffffff')
    neutral_color = param.String(default='#000000')
    accent_base_color = param.String(default='#0072B5')
    collapsed_icon = param.String(default=COLLAPSED_SVG_ICON)
    expanded_icon = param.String(default=EXPANDED_SVG_ICON)
    color = param.String(default='#2B2B2B')
    neutral_fill_card_rest = param.String(default='#F7F7F7')
    neutral_focus = param.String(default='#888888')
    neutral_foreground_rest = param.String(default='#2B2B2B')
    header_luminance = param.Magnitude(default=0.23)
    header_background = param.String(default='#0072B5')
    header_neutral_color = param.String(default='#0072B5')
    header_accent_base_color = param.String(default='#ffffff')
    header_color = param.String(default='#ffffff')
    font = param.String(default='Open Sans, sans-serif')
    font_url = param.String(default=FONT_URL)
    corner_radius = param.Integer(default=3)
    shadow = param.Boolean(default=True)
    luminance = param.Magnitude(default=1.0)

    def create_bokeh_theme(self):
        """Returns a custom bokeh theme based on the style parameters

        Returns:
            Dict: A Bokeh Theme
        """
        return {'attrs': {'figure': {'background_fill_color': self.background_color, 'border_fill_color': self.neutral_fill_card_rest, 'border_fill_alpha': 0, 'outline_line_color': self.neutral_focus, 'outline_line_alpha': 0.5, 'outline_line_width': 1}, 'Grid': {'grid_line_color': self.neutral_focus, 'grid_line_alpha': 0.25}, 'Axis': {'major_tick_line_alpha': 0.5, 'major_tick_line_color': self.neutral_foreground_rest, 'minor_tick_line_alpha': 0.25, 'minor_tick_line_color': self.neutral_foreground_rest, 'axis_line_alpha': 0.1, 'axis_line_color': self.neutral_foreground_rest, 'major_label_text_color': self.neutral_foreground_rest, 'major_label_text_font': self.font, 'major_label_text_font_size': '1.025em', 'axis_label_standoff': 10, 'axis_label_text_color': self.neutral_foreground_rest, 'axis_label_text_font': self.font, 'axis_label_text_font_size': '1.25em', 'axis_label_text_font_style': 'normal'}, 'Legend': {'spacing': 8, 'glyph_width': 15, 'label_standoff': 8, 'label_text_color': self.neutral_foreground_rest, 'label_text_font': self.font, 'label_text_font_size': '1.025em', 'border_line_alpha': 0.5, 'border_line_color': self.neutral_focus, 'background_fill_alpha': 0.25, 'background_fill_color': self.neutral_fill_card_rest}, 'ColorBar': {'background_fill_color': self.background_color, 'title_text_color': self.neutral_foreground_rest, 'title_text_font': self.font, 'title_text_font_size': '1.025em', 'title_text_font_style': 'normal', 'major_label_text_color': self.neutral_foreground_rest, 'major_label_text_font': self.font, 'major_label_text_font_size': '1.025em', 'major_tick_line_alpha': 0, 'bar_line_alpha': 0}, 'Title': {'text_color': self.neutral_foreground_rest, 'text_font': self.font, 'text_font_size': '1.15em'}}}