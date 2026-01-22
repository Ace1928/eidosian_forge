from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, overload
import yaml
from ..core.has_props import HasProps
from ..core.types import PathLike
from ..util.deprecation import deprecated
def _for_class(self, cls: type[Model]) -> dict[str, Any]:
    if cls.__name__ not in self._by_class_cache:
        attrs = self._json['attrs']
        combined: dict[str, Any] = {}
        for base in cls.__mro__[-2::-1]:
            if not issubclass(base, HasProps):
                continue
            self._add_glyph_defaults(base, combined)
            combined.update(attrs.get(base.__name__, _empty_dict))
        if len(combined) == 0:
            combined = _empty_dict
        self._by_class_cache[cls.__name__] = combined
    return self._by_class_cache[cls.__name__]