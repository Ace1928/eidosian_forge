from __future__ import annotations
from copy import deepcopy
from typing import Any, Iterator, List, Mapping, Optional, cast
from attrs import define, field
from fontTools.misc.transform import Transform
from fontTools.pens.basePen import AbstractPen
from fontTools.pens.pointPen import (
from ufoLib2.objects.anchor import Anchor
from ufoLib2.objects.component import Component
from ufoLib2.objects.contour import Contour
from ufoLib2.objects.guideline import Guideline
from ufoLib2.objects.image import Image
from ufoLib2.objects.lib import (
from ufoLib2.objects.misc import BoundingBox, _object_lib, getBounds, getControlBounds
from ufoLib2.pointPens.glyphPointPen import GlyphPointPen
from ufoLib2.serde import serde
from ufoLib2.typing import GlyphSet, HasIdentifier
def appendGuideline(self, guideline: Guideline | Mapping[str, Any]) -> None:
    """Appends a :class:`.Guideline` object to glyph's list of guidelines.

        Args:
            guideline: A :class:`.Guideline` object or a mapping for the Guideline
                constructor.
        """
    if not isinstance(guideline, Guideline):
        if not isinstance(guideline, Mapping):
            raise TypeError('Expected Guideline object or a Mapping for the ', f'Guideline constructor, found {type(guideline).__name__}')
        guideline = Guideline(**guideline)
    self._guidelines.append(guideline)