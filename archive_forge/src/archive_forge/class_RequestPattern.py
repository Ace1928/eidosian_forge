from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
@dataclass
class RequestPattern:
    """
    Request pattern for interception.
    """
    url_pattern: typing.Optional[str] = None
    resource_type: typing.Optional[ResourceType] = None
    interception_stage: typing.Optional[InterceptionStage] = None

    def to_json(self):
        json = dict()
        if self.url_pattern is not None:
            json['urlPattern'] = self.url_pattern
        if self.resource_type is not None:
            json['resourceType'] = self.resource_type.to_json()
        if self.interception_stage is not None:
            json['interceptionStage'] = self.interception_stage.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(url_pattern=str(json['urlPattern']) if 'urlPattern' in json else None, resource_type=ResourceType.from_json(json['resourceType']) if 'resourceType' in json else None, interception_stage=InterceptionStage.from_json(json['interceptionStage']) if 'interceptionStage' in json else None)