from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class ExecutionContextDescription:
    """
    Description of an isolated world.
    """
    id_: ExecutionContextId
    origin: str
    name: str
    aux_data: typing.Optional[dict] = None

    def to_json(self):
        json = dict()
        json['id'] = self.id_.to_json()
        json['origin'] = self.origin
        json['name'] = self.name
        if self.aux_data is not None:
            json['auxData'] = self.aux_data
        return json

    @classmethod
    def from_json(cls, json):
        return cls(id_=ExecutionContextId.from_json(json['id']), origin=str(json['origin']), name=str(json['name']), aux_data=dict(json['auxData']) if 'auxData' in json else None)