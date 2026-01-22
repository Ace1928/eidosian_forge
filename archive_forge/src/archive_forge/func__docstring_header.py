from __future__ import annotations
import logging # isort:skip
from inspect import Parameter
from ..models import Marker
def _docstring_header(glyphclass):
    glyph_class = 'Scatter' if issubclass(glyphclass, Marker) else glyphclass.__name__
    return f'Configure and add :class:`~bokeh.models.glyphs.{glyph_class}` glyphs to this figure.'