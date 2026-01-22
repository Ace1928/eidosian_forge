from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import dom_debugger
from . import page
@dataclass
class LayoutTreeNode:
    """
    Details of an element in the DOM tree with a LayoutObject.
    """
    dom_node_index: int
    bounding_box: dom.Rect
    layout_text: typing.Optional[str] = None
    inline_text_nodes: typing.Optional[typing.List[InlineTextBox]] = None
    style_index: typing.Optional[int] = None
    paint_order: typing.Optional[int] = None
    is_stacking_context: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['domNodeIndex'] = self.dom_node_index
        json['boundingBox'] = self.bounding_box.to_json()
        if self.layout_text is not None:
            json['layoutText'] = self.layout_text
        if self.inline_text_nodes is not None:
            json['inlineTextNodes'] = [i.to_json() for i in self.inline_text_nodes]
        if self.style_index is not None:
            json['styleIndex'] = self.style_index
        if self.paint_order is not None:
            json['paintOrder'] = self.paint_order
        if self.is_stacking_context is not None:
            json['isStackingContext'] = self.is_stacking_context
        return json

    @classmethod
    def from_json(cls, json):
        return cls(dom_node_index=int(json['domNodeIndex']), bounding_box=dom.Rect.from_json(json['boundingBox']), layout_text=str(json['layoutText']) if 'layoutText' in json else None, inline_text_nodes=[InlineTextBox.from_json(i) for i in json['inlineTextNodes']] if 'inlineTextNodes' in json else None, style_index=int(json['styleIndex']) if 'styleIndex' in json else None, paint_order=int(json['paintOrder']) if 'paintOrder' in json else None, is_stacking_context=bool(json['isStackingContext']) if 'isStackingContext' in json else None)