from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from ..models import glyphs
from ..util.deprecation import deprecated
from ._decorators import glyph_method, marker_method
@glyph_method(glyphs.TeXGlyph)
def tex(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
    """
Examples:

    .. code-block:: python

        from bokeh.plotting import figure, show

        p = figure(width=300, height=300)
        p.tex(x=[0], y=[0], text=["x^2 y^2])

        show(p)

"""