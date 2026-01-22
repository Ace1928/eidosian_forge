from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('WebAudio.audioParamWillBeDestroyed')
@dataclass
class AudioParamWillBeDestroyed:
    """
    Notifies that an existing AudioParam has been destroyed.
    """
    context_id: GraphObjectId
    node_id: GraphObjectId
    param_id: GraphObjectId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioParamWillBeDestroyed:
        return cls(context_id=GraphObjectId.from_json(json['contextId']), node_id=GraphObjectId.from_json(json['nodeId']), param_id=GraphObjectId.from_json(json['paramId']))