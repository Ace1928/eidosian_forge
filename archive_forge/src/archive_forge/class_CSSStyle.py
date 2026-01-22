from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class CSSStyle:
    """
    CSS style representation.
    """
    css_properties: typing.List[CSSProperty]
    shorthand_entries: typing.List[ShorthandEntry]
    style_sheet_id: typing.Optional[StyleSheetId] = None
    css_text: typing.Optional[str] = None
    range_: typing.Optional[SourceRange] = None

    def to_json(self):
        json = dict()
        json['cssProperties'] = [i.to_json() for i in self.css_properties]
        json['shorthandEntries'] = [i.to_json() for i in self.shorthand_entries]
        if self.style_sheet_id is not None:
            json['styleSheetId'] = self.style_sheet_id.to_json()
        if self.css_text is not None:
            json['cssText'] = self.css_text
        if self.range_ is not None:
            json['range'] = self.range_.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(css_properties=[CSSProperty.from_json(i) for i in json['cssProperties']], shorthand_entries=[ShorthandEntry.from_json(i) for i in json['shorthandEntries']], style_sheet_id=StyleSheetId.from_json(json['styleSheetId']) if 'styleSheetId' in json else None, css_text=str(json['cssText']) if 'cssText' in json else None, range_=SourceRange.from_json(json['range']) if 'range' in json else None)