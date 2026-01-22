from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import dom_debugger
from . import page
@dataclass
class LayoutTreeSnapshot:
    """
    Table of details of an element in the DOM tree with a LayoutObject.
    """
    node_index: typing.List[int]
    styles: typing.List[ArrayOfStrings]
    bounds: typing.List[Rectangle]
    text: typing.List[StringIndex]
    stacking_contexts: RareBooleanData
    paint_orders: typing.Optional[typing.List[int]] = None
    offset_rects: typing.Optional[typing.List[Rectangle]] = None
    scroll_rects: typing.Optional[typing.List[Rectangle]] = None
    client_rects: typing.Optional[typing.List[Rectangle]] = None

    def to_json(self):
        json = dict()
        json['nodeIndex'] = [i for i in self.node_index]
        json['styles'] = [i.to_json() for i in self.styles]
        json['bounds'] = [i.to_json() for i in self.bounds]
        json['text'] = [i.to_json() for i in self.text]
        json['stackingContexts'] = self.stacking_contexts.to_json()
        if self.paint_orders is not None:
            json['paintOrders'] = [i for i in self.paint_orders]
        if self.offset_rects is not None:
            json['offsetRects'] = [i.to_json() for i in self.offset_rects]
        if self.scroll_rects is not None:
            json['scrollRects'] = [i.to_json() for i in self.scroll_rects]
        if self.client_rects is not None:
            json['clientRects'] = [i.to_json() for i in self.client_rects]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(node_index=[int(i) for i in json['nodeIndex']], styles=[ArrayOfStrings.from_json(i) for i in json['styles']], bounds=[Rectangle.from_json(i) for i in json['bounds']], text=[StringIndex.from_json(i) for i in json['text']], stacking_contexts=RareBooleanData.from_json(json['stackingContexts']), paint_orders=[int(i) for i in json['paintOrders']] if 'paintOrders' in json else None, offset_rects=[Rectangle.from_json(i) for i in json['offsetRects']] if 'offsetRects' in json else None, scroll_rects=[Rectangle.from_json(i) for i in json['scrollRects']] if 'scrollRects' in json else None, client_rects=[Rectangle.from_json(i) for i in json['clientRects']] if 'clientRects' in json else None)