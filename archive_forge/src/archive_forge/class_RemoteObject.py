from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class RemoteObject:
    """
    Mirror object referencing original JavaScript object.
    """
    type_: str
    subtype: typing.Optional[str] = None
    class_name: typing.Optional[str] = None
    value: typing.Optional[typing.Any] = None
    unserializable_value: typing.Optional[UnserializableValue] = None
    description: typing.Optional[str] = None
    object_id: typing.Optional[RemoteObjectId] = None
    preview: typing.Optional[ObjectPreview] = None
    custom_preview: typing.Optional[CustomPreview] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        if self.subtype is not None:
            json['subtype'] = self.subtype
        if self.class_name is not None:
            json['className'] = self.class_name
        if self.value is not None:
            json['value'] = self.value
        if self.unserializable_value is not None:
            json['unserializableValue'] = self.unserializable_value.to_json()
        if self.description is not None:
            json['description'] = self.description
        if self.object_id is not None:
            json['objectId'] = self.object_id.to_json()
        if self.preview is not None:
            json['preview'] = self.preview.to_json()
        if self.custom_preview is not None:
            json['customPreview'] = self.custom_preview.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(type_=str(json['type']), subtype=str(json['subtype']) if 'subtype' in json else None, class_name=str(json['className']) if 'className' in json else None, value=json['value'] if 'value' in json else None, unserializable_value=UnserializableValue.from_json(json['unserializableValue']) if 'unserializableValue' in json else None, description=str(json['description']) if 'description' in json else None, object_id=RemoteObjectId.from_json(json['objectId']) if 'objectId' in json else None, preview=ObjectPreview.from_json(json['preview']) if 'preview' in json else None, custom_preview=CustomPreview.from_json(json['customPreview']) if 'customPreview' in json else None)