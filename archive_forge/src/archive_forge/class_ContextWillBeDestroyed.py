from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('WebAudio.contextWillBeDestroyed')
@dataclass
class ContextWillBeDestroyed:
    """
    Notifies that an existing BaseAudioContext will be destroyed.
    """
    context_id: GraphObjectId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ContextWillBeDestroyed:
        return cls(context_id=GraphObjectId.from_json(json['contextId']))