from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
@dataclass
class ScriptFontFamilies:
    """
    Font families collection for a script.
    """
    script: str
    font_families: FontFamilies

    def to_json(self):
        json = dict()
        json['script'] = self.script
        json['fontFamilies'] = self.font_families.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(script=str(json['script']), font_families=FontFamilies.from_json(json['fontFamilies']))