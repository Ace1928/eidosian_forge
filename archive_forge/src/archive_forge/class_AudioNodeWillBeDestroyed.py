from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('WebAudio.audioNodeWillBeDestroyed')
@dataclass
class AudioNodeWillBeDestroyed:
    """
    Notifies that an existing AudioNode has been destroyed.
    """
    context_id: GraphObjectId
    node_id: GraphObjectId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioNodeWillBeDestroyed:
        return cls(context_id=GraphObjectId.from_json(json['contextId']), node_id=GraphObjectId.from_json(json['nodeId']))