from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class CSSSupports:
    """
    CSS Supports at-rule descriptor.
    """
    text: str
    active: bool
    range_: typing.Optional[SourceRange] = None
    style_sheet_id: typing.Optional[StyleSheetId] = None

    def to_json(self):
        json = dict()
        json['text'] = self.text
        json['active'] = self.active
        if self.range_ is not None:
            json['range'] = self.range_.to_json()
        if self.style_sheet_id is not None:
            json['styleSheetId'] = self.style_sheet_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(text=str(json['text']), active=bool(json['active']), range_=SourceRange.from_json(json['range']) if 'range' in json else None, style_sheet_id=StyleSheetId.from_json(json['styleSheetId']) if 'styleSheetId' in json else None)