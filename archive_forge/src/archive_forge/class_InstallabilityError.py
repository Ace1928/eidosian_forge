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
class InstallabilityError:
    """
    The installability error
    """
    error_id: str
    error_arguments: typing.List[InstallabilityErrorArgument]

    def to_json(self):
        json = dict()
        json['errorId'] = self.error_id
        json['errorArguments'] = [i.to_json() for i in self.error_arguments]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(error_id=str(json['errorId']), error_arguments=[InstallabilityErrorArgument.from_json(i) for i in json['errorArguments']])