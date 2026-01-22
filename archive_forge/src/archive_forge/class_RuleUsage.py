from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class RuleUsage:
    """
    CSS coverage information.
    """
    style_sheet_id: StyleSheetId
    start_offset: float
    end_offset: float
    used: bool

    def to_json(self):
        json = dict()
        json['styleSheetId'] = self.style_sheet_id.to_json()
        json['startOffset'] = self.start_offset
        json['endOffset'] = self.end_offset
        json['used'] = self.used
        return json

    @classmethod
    def from_json(cls, json):
        return cls(style_sheet_id=StyleSheetId.from_json(json['styleSheetId']), start_offset=float(json['startOffset']), end_offset=float(json['endOffset']), used=bool(json['used']))