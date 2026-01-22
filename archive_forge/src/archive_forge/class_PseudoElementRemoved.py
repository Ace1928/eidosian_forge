from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
@event_class('DOM.pseudoElementRemoved')
@dataclass
class PseudoElementRemoved:
    """
    **EXPERIMENTAL**

    Called when a pseudo element is removed from an element.
    """
    parent_id: NodeId
    pseudo_element_id: NodeId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PseudoElementRemoved:
        return cls(parent_id=NodeId.from_json(json['parentId']), pseudo_element_id=NodeId.from_json(json['pseudoElementId']))