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
class FontFamilies:
    """
    Generic font families collection.
    """
    standard: typing.Optional[str] = None
    fixed: typing.Optional[str] = None
    serif: typing.Optional[str] = None
    sans_serif: typing.Optional[str] = None
    cursive: typing.Optional[str] = None
    fantasy: typing.Optional[str] = None
    math: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        if self.standard is not None:
            json['standard'] = self.standard
        if self.fixed is not None:
            json['fixed'] = self.fixed
        if self.serif is not None:
            json['serif'] = self.serif
        if self.sans_serif is not None:
            json['sansSerif'] = self.sans_serif
        if self.cursive is not None:
            json['cursive'] = self.cursive
        if self.fantasy is not None:
            json['fantasy'] = self.fantasy
        if self.math is not None:
            json['math'] = self.math
        return json

    @classmethod
    def from_json(cls, json):
        return cls(standard=str(json['standard']) if 'standard' in json else None, fixed=str(json['fixed']) if 'fixed' in json else None, serif=str(json['serif']) if 'serif' in json else None, sans_serif=str(json['sansSerif']) if 'sansSerif' in json else None, cursive=str(json['cursive']) if 'cursive' in json else None, fantasy=str(json['fantasy']) if 'fantasy' in json else None, math=str(json['math']) if 'math' in json else None)