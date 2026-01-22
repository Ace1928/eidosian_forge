from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class InternalPropertyDescriptor:
    """
    Object internal property descriptor. This property isn't normally visible in JavaScript code.
    """
    name: str
    value: typing.Optional[RemoteObject] = None

    def to_json(self):
        json = dict()
        json['name'] = self.name
        if self.value is not None:
            json['value'] = self.value.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(name=str(json['name']), value=RemoteObject.from_json(json['value']) if 'value' in json else None)