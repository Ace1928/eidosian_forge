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
class AttrDictMixin(AttrDictMixinMapping):
    """Read attribute values using mapping interface.

    For use with Anchors, Guidelines and WoffMetadata classes, where client code
    expects them to behave as dict.
    """

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

    def __getitem__(self, key: str) -> Any:
        attr_name = self._key_to_attr_map()[key]
        try:
            value = getattr(self, attr_name)
        except AttributeError as e:
            raise KeyError(key) from e
        if value is None:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        key_map = self._key_to_attr_map(reverse=True)
        cls = cast('Type[AttrsInstance]', self.__class__)
        for attr_name in attrs.fields_dict(cls):
            if getattr(self, attr_name) is not None:
                yield key_map[attr_name]

    def __len__(self) -> int:
        return sum((1 for _ in self))

    @classmethod
    def coerce_from_dict(cls: Type[_T], value: _T | Mapping[str, Any]) -> _T:
        if isinstance(value, cls):
            return value
        elif isinstance(value, Mapping):
            attr_map = cls._key_to_attr_map()
            return cls(**{attr_map[k]: v for k, v in value.items()})
        raise TypeError(f'Expected {cls.__name__} or mapping, found: {type(value).__name__}')

    @classmethod
    def coerce_from_optional_dict(cls: Type[_T], value: _T | Mapping[str, Any] | None) -> _T | None:
        if value is None:
            return None
        return cls.coerce_from_dict(value)