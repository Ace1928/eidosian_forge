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
def appendAnchor(self, anchor: Anchor | Mapping[str, Any]) -> None:
    """Appends an :class:`.Anchor` object to glyph's list of anchors.

        Args:
            anchor: An :class:`.Anchor` object or mapping for the Anchor constructor.
        """
    if not isinstance(anchor, Anchor):
        if not isinstance(anchor, Mapping):
            raise TypeError('Expected Anchor object or a Mapping for the ', f'Anchor constructor, found {type(anchor).__name__}')
        anchor = Anchor(**anchor)
    self.anchors.append(anchor)