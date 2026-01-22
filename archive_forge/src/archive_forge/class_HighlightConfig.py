from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
@dataclass
class HighlightConfig:
    """
    Configuration data for the highlighting of page elements.
    """
    show_info: typing.Optional[bool] = None
    show_styles: typing.Optional[bool] = None
    show_rulers: typing.Optional[bool] = None
    show_accessibility_info: typing.Optional[bool] = None
    show_extension_lines: typing.Optional[bool] = None
    content_color: typing.Optional[dom.RGBA] = None
    padding_color: typing.Optional[dom.RGBA] = None
    border_color: typing.Optional[dom.RGBA] = None
    margin_color: typing.Optional[dom.RGBA] = None
    event_target_color: typing.Optional[dom.RGBA] = None
    shape_color: typing.Optional[dom.RGBA] = None
    shape_margin_color: typing.Optional[dom.RGBA] = None
    css_grid_color: typing.Optional[dom.RGBA] = None
    color_format: typing.Optional[ColorFormat] = None
    grid_highlight_config: typing.Optional[GridHighlightConfig] = None

    def to_json(self):
        json = dict()
        if self.show_info is not None:
            json['showInfo'] = self.show_info
        if self.show_styles is not None:
            json['showStyles'] = self.show_styles
        if self.show_rulers is not None:
            json['showRulers'] = self.show_rulers
        if self.show_accessibility_info is not None:
            json['showAccessibilityInfo'] = self.show_accessibility_info
        if self.show_extension_lines is not None:
            json['showExtensionLines'] = self.show_extension_lines
        if self.content_color is not None:
            json['contentColor'] = self.content_color.to_json()
        if self.padding_color is not None:
            json['paddingColor'] = self.padding_color.to_json()
        if self.border_color is not None:
            json['borderColor'] = self.border_color.to_json()
        if self.margin_color is not None:
            json['marginColor'] = self.margin_color.to_json()
        if self.event_target_color is not None:
            json['eventTargetColor'] = self.event_target_color.to_json()
        if self.shape_color is not None:
            json['shapeColor'] = self.shape_color.to_json()
        if self.shape_margin_color is not None:
            json['shapeMarginColor'] = self.shape_margin_color.to_json()
        if self.css_grid_color is not None:
            json['cssGridColor'] = self.css_grid_color.to_json()
        if self.color_format is not None:
            json['colorFormat'] = self.color_format.to_json()
        if self.grid_highlight_config is not None:
            json['gridHighlightConfig'] = self.grid_highlight_config.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(show_info=bool(json['showInfo']) if 'showInfo' in json else None, show_styles=bool(json['showStyles']) if 'showStyles' in json else None, show_rulers=bool(json['showRulers']) if 'showRulers' in json else None, show_accessibility_info=bool(json['showAccessibilityInfo']) if 'showAccessibilityInfo' in json else None, show_extension_lines=bool(json['showExtensionLines']) if 'showExtensionLines' in json else None, content_color=dom.RGBA.from_json(json['contentColor']) if 'contentColor' in json else None, padding_color=dom.RGBA.from_json(json['paddingColor']) if 'paddingColor' in json else None, border_color=dom.RGBA.from_json(json['borderColor']) if 'borderColor' in json else None, margin_color=dom.RGBA.from_json(json['marginColor']) if 'marginColor' in json else None, event_target_color=dom.RGBA.from_json(json['eventTargetColor']) if 'eventTargetColor' in json else None, shape_color=dom.RGBA.from_json(json['shapeColor']) if 'shapeColor' in json else None, shape_margin_color=dom.RGBA.from_json(json['shapeMarginColor']) if 'shapeMarginColor' in json else None, css_grid_color=dom.RGBA.from_json(json['cssGridColor']) if 'cssGridColor' in json else None, color_format=ColorFormat.from_json(json['colorFormat']) if 'colorFormat' in json else None, grid_highlight_config=GridHighlightConfig.from_json(json['gridHighlightConfig']) if 'gridHighlightConfig' in json else None)