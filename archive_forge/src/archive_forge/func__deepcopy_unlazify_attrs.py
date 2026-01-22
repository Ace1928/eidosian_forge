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
def _deepcopy_unlazify_attrs(self: Any, memo: Any) -> Any:
    if self._lazy:
        self.unlazify()
    return self.__class__(**{a.name if a.name[0] != '_' else a.name[1:]: deepcopy(getattr(self, a.name), memo) for a in attrs.fields(self.__class__) if a.init})