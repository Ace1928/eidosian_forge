from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
@event_class('DOM.characterDataModified')
@dataclass
class CharacterDataModified:
    """
    Mirrors ``DOMCharacterDataModified`` event.
    """
    node_id: NodeId
    character_data: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CharacterDataModified:
        return cls(node_id=NodeId.from_json(json['nodeId']), character_data=str(json['characterData']))