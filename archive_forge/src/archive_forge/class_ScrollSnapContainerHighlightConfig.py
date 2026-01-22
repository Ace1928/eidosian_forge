from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
@dataclass
class ScrollSnapContainerHighlightConfig:
    snapport_border: typing.Optional[LineStyle] = None
    snap_area_border: typing.Optional[LineStyle] = None
    scroll_margin_color: typing.Optional[dom.RGBA] = None
    scroll_padding_color: typing.Optional[dom.RGBA] = None

    def to_json(self):
        json = dict()
        if self.snapport_border is not None:
            json['snapportBorder'] = self.snapport_border.to_json()
        if self.snap_area_border is not None:
            json['snapAreaBorder'] = self.snap_area_border.to_json()
        if self.scroll_margin_color is not None:
            json['scrollMarginColor'] = self.scroll_margin_color.to_json()
        if self.scroll_padding_color is not None:
            json['scrollPaddingColor'] = self.scroll_padding_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(snapport_border=LineStyle.from_json(json['snapportBorder']) if 'snapportBorder' in json else None, snap_area_border=LineStyle.from_json(json['snapAreaBorder']) if 'snapAreaBorder' in json else None, scroll_margin_color=dom.RGBA.from_json(json['scrollMarginColor']) if 'scrollMarginColor' in json else None, scroll_padding_color=dom.RGBA.from_json(json['scrollPaddingColor']) if 'scrollPaddingColor' in json else None)