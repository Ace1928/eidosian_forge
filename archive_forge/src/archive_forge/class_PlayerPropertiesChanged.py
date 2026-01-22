from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('Media.playerPropertiesChanged')
@dataclass
class PlayerPropertiesChanged:
    """
    This can be called multiple times, and can be used to set / override /
    remove player properties. A null propValue indicates removal.
    """
    player_id: PlayerId
    properties: typing.List[PlayerProperty]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PlayerPropertiesChanged:
        return cls(player_id=PlayerId.from_json(json['playerId']), properties=[PlayerProperty.from_json(i) for i in json['properties']])