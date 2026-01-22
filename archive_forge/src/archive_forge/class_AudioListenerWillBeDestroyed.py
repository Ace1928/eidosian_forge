from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('WebAudio.audioListenerWillBeDestroyed')
@dataclass
class AudioListenerWillBeDestroyed:
    """
    Notifies that a new AudioListener has been created.
    """
    context_id: GraphObjectId
    listener_id: GraphObjectId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AudioListenerWillBeDestroyed:
        return cls(context_id=GraphObjectId.from_json(json['contextId']), listener_id=GraphObjectId.from_json(json['listenerId']))