from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class MathMLGlyph(MathTextGlyph):
    """ Render mathematical content using `MathML <https://www.w3.org/Math/>`_
    notation.

    See :ref:`ug_styling_mathtext` in the |user guide| for more information.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)