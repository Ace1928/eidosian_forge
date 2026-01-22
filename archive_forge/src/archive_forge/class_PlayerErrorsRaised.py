from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('Media.playerErrorsRaised')
@dataclass
class PlayerErrorsRaised:
    """
    Send a list of any errors that need to be delivered.
    """
    player_id: PlayerId
    errors: typing.List[PlayerError]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PlayerErrorsRaised:
        return cls(player_id=PlayerId.from_json(json['playerId']), errors=[PlayerError.from_json(i) for i in json['errors']])