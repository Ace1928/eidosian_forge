from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import runtime
@event_class('DOM.topLayerElementsUpdated')
@dataclass
class TopLayerElementsUpdated:
    """
    **EXPERIMENTAL**

    Called when top layer elements are changed.
    """

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TopLayerElementsUpdated:
        return cls()