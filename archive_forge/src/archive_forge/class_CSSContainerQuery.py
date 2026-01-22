from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
@dataclass
class CSSContainerQuery:
    """
    CSS container query rule descriptor.
    """
    text: str
    range_: typing.Optional[SourceRange] = None
    style_sheet_id: typing.Optional[StyleSheetId] = None
    name: typing.Optional[str] = None
    physical_axes: typing.Optional[dom.PhysicalAxes] = None
    logical_axes: typing.Optional[dom.LogicalAxes] = None

    def to_json(self):
        json = dict()
        json['text'] = self.text
        if self.range_ is not None:
            json['range'] = self.range_.to_json()
        if self.style_sheet_id is not None:
            json['styleSheetId'] = self.style_sheet_id.to_json()
        if self.name is not None:
            json['name'] = self.name
        if self.physical_axes is not None:
            json['physicalAxes'] = self.physical_axes.to_json()
        if self.logical_axes is not None:
            json['logicalAxes'] = self.logical_axes.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(text=str(json['text']), range_=SourceRange.from_json(json['range']) if 'range' in json else None, style_sheet_id=StyleSheetId.from_json(json['styleSheetId']) if 'styleSheetId' in json else None, name=str(json['name']) if 'name' in json else None, physical_axes=dom.PhysicalAxes.from_json(json['physicalAxes']) if 'physicalAxes' in json else None, logical_axes=dom.LogicalAxes.from_json(json['logicalAxes']) if 'logicalAxes' in json else None)