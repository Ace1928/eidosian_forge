from collections import namedtuple
from fontTools.misc import sstruct
from fontTools import ttLib
from fontTools import version
from fontTools.misc.transform import DecomposedTransform
from fontTools.misc.textTools import tostr, safeEval, pad
from fontTools.misc.arrayTools import updateBounds, pointInRect
from fontTools.misc.bezierTools import calcQuadraticBounds
from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.vector import Vector
from numbers import Number
from . import DefaultTable
from . import ttProgram
import sys
import struct
import array
import logging
import math
import os
from fontTools.misc import xmlWriter
from fontTools.misc.filenames import userNameToFileName
from fontTools.misc.loggingTools import deprecateFunction
from enum import IntFlag
from functools import partial
from types import SimpleNamespace
from typing import Set
def _getCoordinatesAndControls(self, glyphName, hMetrics, vMetrics=None, *, round=otRound):
    """Return glyph coordinates and controls as expected by "gvar" table.

        The coordinates includes four "phantom points" for the glyph metrics,
        as mandated by the "gvar" spec.

        The glyph controls is a namedtuple with the following attributes:
                - numberOfContours: -1 for composite glyphs.
                - endPts: list of indices of end points for each contour in simple
                glyphs, or component indices in composite glyphs (used for IUP
                optimization).
                - flags: array of contour point flags for simple glyphs (None for
                composite glyphs).
                - components: list of base glyph names (str) for each component in
                composite glyphs (None for simple glyphs).

        The "hMetrics" and vMetrics are used to compute the "phantom points" (see
        the "_getPhantomPoints" method).

        Return None if the requested glyphName is not present.
        """
    glyph = self.get(glyphName)
    if glyph is None:
        return None
    if glyph.isComposite():
        coords = GlyphCoordinates([(getattr(c, 'x', 0), getattr(c, 'y', 0)) for c in glyph.components])
        controls = _GlyphControls(numberOfContours=glyph.numberOfContours, endPts=list(range(len(glyph.components))), flags=None, components=[(c.glyphName, getattr(c, 'transform', None)) for c in glyph.components])
    elif glyph.isVarComposite():
        coords = []
        controls = []
        for component in glyph.components:
            componentCoords, componentControls = component.getCoordinatesAndControls()
            coords.extend(componentCoords)
            controls.extend(componentControls)
        coords = GlyphCoordinates(coords)
        controls = _GlyphControls(numberOfContours=glyph.numberOfContours, endPts=list(range(len(coords))), flags=None, components=[(c.glyphName, getattr(c, 'flags', None)) for c in glyph.components])
    else:
        coords, endPts, flags = glyph.getCoordinates(self)
        coords = coords.copy()
        controls = _GlyphControls(numberOfContours=glyph.numberOfContours, endPts=endPts, flags=flags, components=None)
    phantomPoints = self._getPhantomPoints(glyphName, hMetrics, vMetrics)
    coords.extend(phantomPoints)
    coords.toInt(round=round)
    return (coords, controls)