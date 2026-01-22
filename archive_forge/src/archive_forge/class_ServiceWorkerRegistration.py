from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import target
@dataclass
class ServiceWorkerRegistration:
    """
    ServiceWorker registration.
    """
    registration_id: RegistrationID
    scope_url: str
    is_deleted: bool

    def to_json(self):
        json = dict()
        json['registrationId'] = self.registration_id.to_json()
        json['scopeURL'] = self.scope_url
        json['isDeleted'] = self.is_deleted
        return json

    @classmethod
    def from_json(cls, json):
        return cls(registration_id=RegistrationID.from_json(json['registrationId']), scope_url=str(json['scopeURL']), is_deleted=bool(json['isDeleted']))