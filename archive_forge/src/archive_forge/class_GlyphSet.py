from __future__ import annotations
import os
import sys
from typing import TypeVar, Union
from fontTools.pens.basePen import AbstractPen
from fontTools.pens.pointPen import AbstractPointPen
class GlyphSet(Protocol):
    """Any container that holds drawable objects.

    In ufoLib2, this usually refers to :class:`.Font` (referencing glyphs in the
    default layer) and :class:`.Layer` (referencing glyphs in that particular layer).
    Ideally, this would be a simple subclass of ``Mapping[str, Union[Drawable, DrawablePoints]]``,
    but due to historic reasons, the established objects don't conform to ``Mapping``
    exactly.

    The protocol contains what is used in :mod:`fontTools.pens` at v4.18.2
    (grep for ``.glyphSet``).
    """

    def __contains__(self, name: object) -> bool:
        ...

    def __getitem__(self, name: str) -> Drawable | DrawablePoints:
        ...