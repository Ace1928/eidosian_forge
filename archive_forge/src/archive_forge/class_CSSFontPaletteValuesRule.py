from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class CSSFontPaletteValuesRule:
    """
    CSS font-palette-values rule representation.
    """
    origin: StyleSheetOrigin
    font_palette_name: Value
    style: CSSStyle
    style_sheet_id: typing.Optional[StyleSheetId] = None

    def to_json(self):
        json = dict()
        json['origin'] = self.origin.to_json()
        json['fontPaletteName'] = self.font_palette_name.to_json()
        json['style'] = self.style.to_json()
        if self.style_sheet_id is not None:
            json['styleSheetId'] = self.style_sheet_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(origin=StyleSheetOrigin.from_json(json['origin']), font_palette_name=Value.from_json(json['fontPaletteName']), style=CSSStyle.from_json(json['style']), style_sheet_id=StyleSheetId.from_json(json['styleSheetId']) if 'styleSheetId' in json else None)