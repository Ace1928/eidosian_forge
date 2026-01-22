from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
@dataclass
class IsolationModeHighlightConfig:
    resizer_color: typing.Optional[dom.RGBA] = None
    resizer_handle_color: typing.Optional[dom.RGBA] = None
    mask_color: typing.Optional[dom.RGBA] = None

    def to_json(self):
        json = dict()
        if self.resizer_color is not None:
            json['resizerColor'] = self.resizer_color.to_json()
        if self.resizer_handle_color is not None:
            json['resizerHandleColor'] = self.resizer_handle_color.to_json()
        if self.mask_color is not None:
            json['maskColor'] = self.mask_color.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(resizer_color=dom.RGBA.from_json(json['resizerColor']) if 'resizerColor' in json else None, resizer_handle_color=dom.RGBA.from_json(json['resizerHandleColor']) if 'resizerHandleColor' in json else None, mask_color=dom.RGBA.from_json(json['maskColor']) if 'maskColor' in json else None)