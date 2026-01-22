from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any, overload
import yaml
from ..core.has_props import HasProps
from ..core.types import PathLike
from ..util.deprecation import deprecated
def _add_glyph_defaults(self, cls: type[HasProps], props: dict[str, Any]) -> None:
    from ..models.glyphs import Glyph
    if issubclass(cls, Glyph):
        if hasattr(cls, 'line_alpha'):
            props.update(self._line_defaults)
        if hasattr(cls, 'fill_alpha'):
            props.update(self._fill_defaults)
        if hasattr(cls, 'text_alpha'):
            props.update(self._text_defaults)