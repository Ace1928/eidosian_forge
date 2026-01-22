from __future__ import annotations
import collections.abc
import uuid
from abc import abstractmethod
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from functools import lru_cache
from typing import (
import attrs
from attrs import define, field
from fontTools.misc.arrayTools import unionRect
from fontTools.misc.transform import Transform
from fontTools.pens.boundsPen import BoundsPen, ControlBoundsPen
from fontTools.ufoLib import UFOReader, UFOWriter
from ufoLib2.constants import OBJECT_LIBS_KEY
from ufoLib2.typing import Drawable, GlyphSet, HasIdentifier
@classmethod
@lru_cache(maxsize=None)
def _key_to_attr_map(cls: Type[AttrsInstance], reverse: bool=False) -> dict[str, str]:
    result = {}
    for a in attrs.fields(cls):
        attr_name = a.name
        key = attr_name
        if 'rename_attr' in a.metadata:
            key = a.metadata['rename_attr']
        if reverse:
            result[attr_name] = key
        else:
            result[key] = attr_name
    return result