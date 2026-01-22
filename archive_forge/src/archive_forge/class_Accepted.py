from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('Tethering.accepted')
@dataclass
class Accepted:
    """
    Informs that port was successfully bound and got a specified connection id.
    """
    port: int
    connection_id: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> Accepted:
        return cls(port=int(json['port']), connection_id=str(json['connectionId']))