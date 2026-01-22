from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import runtime
@dataclass
class ObjectStore:
    """
    Object store.
    """
    name: str
    key_path: KeyPath
    auto_increment: bool
    indexes: typing.List[ObjectStoreIndex]

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['keyPath'] = self.key_path.to_json()
        json['autoIncrement'] = self.auto_increment
        json['indexes'] = [i.to_json() for i in self.indexes]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(name=str(json['name']), key_path=KeyPath.from_json(json['keyPath']), auto_increment=bool(json['autoIncrement']), indexes=[ObjectStoreIndex.from_json(i) for i in json['indexes']])