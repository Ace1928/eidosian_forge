from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from ..models import glyphs
from ..util.deprecation import deprecated
from ._decorators import glyph_method, marker_method
@glyph_method(glyphs.VStrip)
def vstrip(self, *args: Any, **kwargs: Any) -> GlyphRenderer:
    """
Examples:

    .. code-block:: python

        from bokeh.plotting import figure, output_file, show

        plot = figure(width=300, height=300, y_range=(0, 1))
        plot.vstrip(x0=[1, 2, 5], x1=[3, 4, 8], color="#CAB2D6")

        show(plot)

"""