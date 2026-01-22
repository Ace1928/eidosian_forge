from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class PlatformFontUsage:
    """
    Information about amount of glyphs that were rendered with given font.
    """
    family_name: str
    post_script_name: str
    is_custom_font: bool
    glyph_count: float

    def to_json(self):
        json = dict()
        json['familyName'] = self.family_name
        json['postScriptName'] = self.post_script_name
        json['isCustomFont'] = self.is_custom_font
        json['glyphCount'] = self.glyph_count
        return json

    @classmethod
    def from_json(cls, json):
        return cls(family_name=str(json['familyName']), post_script_name=str(json['postScriptName']), is_custom_font=bool(json['isCustomFont']), glyph_count=float(json['glyphCount']))