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
def _setCoordinates(self, glyphName, coord, hMetrics, vMetrics=None):
    """Set coordinates and metrics for the given glyph.

        "coord" is an array of GlyphCoordinates which must include the "phantom
        points" as the last four coordinates.

        Both the horizontal/vertical advances and left/top sidebearings in "hmtx"
        and "vmtx" tables (if any) are updated from four phantom points and
        the glyph's bounding boxes.

        The "hMetrics" and vMetrics are used to propagate "phantom points"
        into "hmtx" and "vmtx" tables if desired.  (see the "_getPhantomPoints"
        method).
        """
    glyph = self[glyphName]
    assert len(coord) >= 4
    leftSideX = coord[-4][0]
    rightSideX = coord[-3][0]
    topSideY = coord[-2][1]
    bottomSideY = coord[-1][1]
    coord = coord[:-4]
    if glyph.isComposite():
        assert len(coord) == len(glyph.components)
        for p, comp in zip(coord, glyph.components):
            if hasattr(comp, 'x'):
                comp.x, comp.y = p
    elif glyph.isVarComposite():
        for comp in glyph.components:
            coord = comp.setCoordinates(coord)
        assert not coord
    elif glyph.numberOfContours == 0:
        assert len(coord) == 0
    else:
        assert len(coord) == len(glyph.coordinates)
        glyph.coordinates = GlyphCoordinates(coord)
    glyph.recalcBounds(self, boundsDone=set())
    horizontalAdvanceWidth = otRound(rightSideX - leftSideX)
    if horizontalAdvanceWidth < 0:
        horizontalAdvanceWidth = 0
    leftSideBearing = otRound(glyph.xMin - leftSideX)
    hMetrics[glyphName] = (horizontalAdvanceWidth, leftSideBearing)
    if vMetrics is not None:
        verticalAdvanceWidth = otRound(topSideY - bottomSideY)
        if verticalAdvanceWidth < 0:
            verticalAdvanceWidth = 0
        topSideBearing = otRound(topSideY - glyph.yMax)
        vMetrics[glyphName] = (verticalAdvanceWidth, topSideBearing)