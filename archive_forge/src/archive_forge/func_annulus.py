from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from ..models import glyphs
from ..util.deprecation import deprecated
from ._decorators import glyph_method, marker_method
@glyph_method(glyphs.Annulus)
def annulus(self, **kwargs: Any) -> GlyphRenderer:
    """
Examples:

    .. code-block:: python

        from bokeh.plotting import figure, output_file, show

        plot = figure(width=300, height=300)
        plot.annulus(x=[1, 2, 3], y=[1, 2, 3], color="#7FC97F",
                     inner_radius=0.2, outer_radius=0.5)

        show(plot)

"""