from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
from . import runtime
@dataclass
class DeprecationIssueDetails:
    """
    This issue tracks information needed to print a deprecation message.
    https://source.chromium.org/chromium/chromium/src/+/main:third_party/blink/renderer/core/frame/third_party/blink/renderer/core/frame/deprecation/README.md
    """
    source_code_location: SourceCodeLocation
    type_: str
    affected_frame: typing.Optional[AffectedFrame] = None

    def to_json(self):
        json = dict()
        json['sourceCodeLocation'] = self.source_code_location.to_json()
        json['type'] = self.type_
        if self.affected_frame is not None:
            json['affectedFrame'] = self.affected_frame.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(source_code_location=SourceCodeLocation.from_json(json['sourceCodeLocation']), type_=str(json['type']), affected_frame=AffectedFrame.from_json(json['affectedFrame']) if 'affectedFrame' in json else None)