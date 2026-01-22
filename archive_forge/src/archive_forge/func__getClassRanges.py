import copy
from enum import IntEnum
from functools import reduce
from math import radians
import itertools
from collections import defaultdict, namedtuple
from fontTools.ttLib.tables.otTraverse import dfs_base_table
from fontTools.misc.arrayTools import quantizeRect
from fontTools.misc.roundTools import otRound
from fontTools.misc.transform import Transform, Identity
from fontTools.misc.textTools import bytesjoin, pad, safeEval
from fontTools.pens.boundsPen import ControlBoundsPen
from fontTools.pens.transformPen import TransformPen
from .otBase import (
from fontTools.feaLib.lookupDebugInfo import LookupDebugInfo, LOOKUP_DEBUG_INFO_KEY
import logging
import struct
from typing import TYPE_CHECKING, Iterator, List, Optional, Set
def _getClassRanges(self, font):
    classDefs = getattr(self, 'classDefs', None)
    if classDefs is None:
        self.classDefs = {}
        return
    getGlyphID = font.getGlyphID
    items = []
    for glyphName, cls in classDefs.items():
        if not cls:
            continue
        items.append((getGlyphID(glyphName), glyphName, cls))
    if items:
        items.sort()
        last, lastName, lastCls = items[0]
        ranges = [[lastCls, last, lastName]]
        for glyphID, glyphName, cls in items[1:]:
            if glyphID != last + 1 or cls != lastCls:
                ranges[-1].extend([last, lastName])
                ranges.append([cls, glyphID, glyphName])
            last = glyphID
            lastName = glyphName
            lastCls = cls
        ranges[-1].extend([last, lastName])
        return ranges