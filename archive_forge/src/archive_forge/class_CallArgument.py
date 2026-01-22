from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class CallArgument:
    """
    Represents function call argument. Either remote object id ``objectId``, primitive ``value``,
    unserializable primitive value or neither of (for undefined) them should be specified.
    """
    value: typing.Optional[typing.Any] = None
    unserializable_value: typing.Optional[UnserializableValue] = None
    object_id: typing.Optional[RemoteObjectId] = None

    def to_json(self):
        json = dict()
        if self.value is not None:
            json['value'] = self.value
        if self.unserializable_value is not None:
            json['unserializableValue'] = self.unserializable_value.to_json()
        if self.object_id is not None:
            json['objectId'] = self.object_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(value=json['value'] if 'value' in json else None, unserializable_value=UnserializableValue.from_json(json['unserializableValue']) if 'unserializableValue' in json else None, object_id=RemoteObjectId.from_json(json['objectId']) if 'objectId' in json else None)