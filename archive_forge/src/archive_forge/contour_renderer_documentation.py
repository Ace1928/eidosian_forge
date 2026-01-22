from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING
from ...core.properties import Float, Instance, Seq
from .glyph_renderer import GlyphRenderer
from .renderer import DataRenderer
 Construct and return a new ``ContourColorBar`` for this ``ContourRenderer``.

        The color bar will use the same fill, hatch and line visual properties
        as the ContourRenderer. Extra keyword arguments may be passed in to
        control ``BaseColorBar`` properties such as `title`.
        