from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import logging
from fontTools.designspaceLib import (
def _getRibbiStyle(self: DesignSpaceDocument, userLocation: SimpleLocationDict) -> Tuple[RibbiStyle, SimpleLocationDict]:
    """Compute the RIBBI style name of the given user location,
    return the location of the matching Regular in the RIBBI group.

    .. versionadded:: 5.0
    """
    regularUserLocation = {}
    axes_by_tag = {axis.tag: axis for axis in self.axes}
    bold: bool = False
    italic: bool = False
    axis = axes_by_tag.get('wght')
    if axis is not None:
        for regular_label in axis.axisLabels:
            if regular_label.linkedUserValue == userLocation[axis.name] and regular_label.userValue < regular_label.linkedUserValue:
                regularUserLocation[axis.name] = regular_label.userValue
                bold = True
                break
    axis = axes_by_tag.get('ital') or axes_by_tag.get('slnt')
    if axis is not None:
        for upright_label in axis.axisLabels:
            if upright_label.linkedUserValue == userLocation[axis.name] and abs(upright_label.userValue) < abs(upright_label.linkedUserValue):
                regularUserLocation[axis.name] = upright_label.userValue
                italic = True
                break
    return (BOLD_ITALIC_TO_RIBBI_STYLE[bold, italic], {**userLocation, **regularUserLocation})