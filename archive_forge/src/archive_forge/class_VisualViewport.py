from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
@dataclass
class VisualViewport:
    """
    Visual viewport position, dimensions, and scale.
    """
    offset_x: float
    offset_y: float
    page_x: float
    page_y: float
    client_width: float
    client_height: float
    scale: float
    zoom: typing.Optional[float] = None

    def to_json(self):
        json = dict()
        json['offsetX'] = self.offset_x
        json['offsetY'] = self.offset_y
        json['pageX'] = self.page_x
        json['pageY'] = self.page_y
        json['clientWidth'] = self.client_width
        json['clientHeight'] = self.client_height
        json['scale'] = self.scale
        if self.zoom is not None:
            json['zoom'] = self.zoom
        return json

    @classmethod
    def from_json(cls, json):
        return cls(offset_x=float(json['offsetX']), offset_y=float(json['offsetY']), page_x=float(json['pageX']), page_y=float(json['pageY']), client_width=float(json['clientWidth']), client_height=float(json['clientHeight']), scale=float(json['scale']), zoom=float(json['zoom']) if 'zoom' in json else None)