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
class PermissionsPolicyFeatureState:
    feature: PermissionsPolicyFeature
    allowed: bool
    locator: typing.Optional[PermissionsPolicyBlockLocator] = None

    def to_json(self):
        json = dict()
        json['feature'] = self.feature.to_json()
        json['allowed'] = self.allowed
        if self.locator is not None:
            json['locator'] = self.locator.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(feature=PermissionsPolicyFeature.from_json(json['feature']), allowed=bool(json['allowed']), locator=PermissionsPolicyBlockLocator.from_json(json['locator']) if 'locator' in json else None)