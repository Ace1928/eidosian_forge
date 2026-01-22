from __future__ import annotations
import logging # isort:skip
from typing import ClassVar, Iterator
from .color import RGB
class _ColorGroupMeta(type):
    """ This metaclass enables ColorGroup class types to be used like simple
    enumerations.

    """
    _colors: tuple[str, ...]

    def __len__(self) -> int:
        return len(self._colors)

    def __getitem__(self, v: str | int) -> NamedColor:
        from . import named
        if isinstance(v, str):
            if v in self._colors:
                return getattr(named, v.lower())
            raise KeyError(f'Color group {self.__class__.__name__!r} has no color {v!r}')
        if isinstance(v, int):
            if v >= 0 and v < len(self):
                return getattr(named, self._colors[v].lower())
            raise IndexError(f'Index out of range for color group {self.__class__.__name__!r}')
        raise ValueError(f'Unknown index {v!r} for color group {self.__class__.__name__!r}')

    def __iter__(self) -> Iterator[NamedColor]:
        from . import named
        return (getattr(named, x.lower()) for x in self._colors)

    def __getattr__(self, v: str) -> NamedColor:
        from . import named
        if v != '_colors' and v in self._colors:
            return getattr(named, v.lower())
        return getattr(type, v)