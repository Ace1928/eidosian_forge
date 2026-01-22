from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
@dataclass
class AXRelatedNode:
    backend_dom_node_id: dom.BackendNodeId
    idref: typing.Optional[str] = None
    text: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['backendDOMNodeId'] = self.backend_dom_node_id.to_json()
        if self.idref is not None:
            json['idref'] = self.idref
        if self.text is not None:
            json['text'] = self.text
        return json

    @classmethod
    def from_json(cls, json):
        return cls(backend_dom_node_id=dom.BackendNodeId.from_json(json['backendDOMNodeId']), idref=str(json['idref']) if 'idref' in json else None, text=str(json['text']) if 'text' in json else None)