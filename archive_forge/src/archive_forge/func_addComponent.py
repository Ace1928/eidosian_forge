from __future__ import annotations
from typing import TYPE_CHECKING, Any
from fontTools.misc.transform import Transform
from fontTools.pens.pointPen import AbstractPointPen
from ufoLib2.objects.component import Component
from ufoLib2.objects.contour import Contour
from ufoLib2.objects.point import Point
def addComponent(self, baseGlyph: str, transformation: Transform, identifier: str | None=None, **kwargs: Any) -> None:
    component = Component(baseGlyph, transformation, identifier=identifier)
    self._glyph.components.append(component)