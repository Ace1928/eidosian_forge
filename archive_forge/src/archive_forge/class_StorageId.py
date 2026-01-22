from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class StorageId:
    """
    DOM Storage identifier.
    """
    security_origin: str
    is_local_storage: bool

    def to_json(self):
        json = dict()
        json['securityOrigin'] = self.security_origin
        json['isLocalStorage'] = self.is_local_storage
        return json

    @classmethod
    def from_json(cls, json):
        return cls(security_origin=str(json['securityOrigin']), is_local_storage=bool(json['isLocalStorage']))