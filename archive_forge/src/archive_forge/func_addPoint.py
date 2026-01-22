from __future__ import annotations
from typing import TYPE_CHECKING, Any
from fontTools.misc.transform import Transform
from fontTools.pens.pointPen import AbstractPointPen
from ufoLib2.objects.component import Component
from ufoLib2.objects.contour import Contour
from ufoLib2.objects.point import Point
def addPoint(self, pt: tuple[float, float], segmentType: str | None=None, smooth: bool=False, name: str | None=None, identifier: str | None=None, **kwargs: Any) -> None:
    if self._contour is None:
        raise ValueError('Call beginPath first.')
    x, y = pt
    self._contour.append(Point(x, y, type=segmentType, smooth=smooth, name=name, identifier=identifier))