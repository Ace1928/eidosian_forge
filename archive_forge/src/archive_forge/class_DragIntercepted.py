from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('Input.dragIntercepted')
@dataclass
class DragIntercepted:
    """
    **EXPERIMENTAL**

    Emitted only when ``Input.setInterceptDrags`` is enabled. Use this data with ``Input.dispatchDragEvent`` to
    restore normal drag and drop behavior.
    """
    data: DragData

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DragIntercepted:
        return cls(data=DragData.from_json(json['data']))