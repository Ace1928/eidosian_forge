from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class StyleDeclarationEdit:
    """
    A descriptor of operation to mutate style declaration text.
    """
    style_sheet_id: StyleSheetId
    range_: SourceRange
    text: str

    def to_json(self):
        json = dict()
        json['styleSheetId'] = self.style_sheet_id.to_json()
        json['range'] = self.range_.to_json()
        json['text'] = self.text
        return json

    @classmethod
    def from_json(cls, json):
        return cls(style_sheet_id=StyleSheetId.from_json(json['styleSheetId']), range_=SourceRange.from_json(json['range']), text=str(json['text']))