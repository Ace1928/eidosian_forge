from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
@event_class('LayerTree.layerTreeDidChange')
@dataclass
class LayerTreeDidChange:
    layers: typing.Optional[typing.List[Layer]]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LayerTreeDidChange:
        return cls(layers=[Layer.from_json(i) for i in json['layers']] if 'layers' in json else None)