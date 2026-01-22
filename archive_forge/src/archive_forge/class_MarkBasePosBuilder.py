from collections import namedtuple, OrderedDict
import os
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import (
from fontTools.ttLib.tables import otBase
from fontTools.feaLib.ast import STATNameStatement
from fontTools.otlLib.optimize.gpos import (
from fontTools.otlLib.error import OpenTypeLibError
from functools import reduce
import logging
import copy
class MarkBasePosBuilder(LookupBuilder):
    """Builds a Mark-To-Base Positioning (GPOS4) lookup.

    Users are expected to manually add marks and bases to the ``marks``
    and ``bases`` attributes after the object has been initialized, e.g.::

        builder.marks["acute"]   = (0, a1)
        builder.marks["grave"]   = (0, a1)
        builder.marks["cedilla"] = (1, a2)
        builder.bases["a"] = {0: a3, 1: a5}
        builder.bases["b"] = {0: a4, 1: a5}

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        marks: An dictionary mapping a glyph name to a two-element
            tuple containing a mark class ID and ``otTables.Anchor`` object.
        bases: An dictionary mapping a glyph name to a dictionary of
            mark class IDs and ``otTables.Anchor`` object.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, 'GPOS', 4)
        self.marks = {}
        self.bases = {}

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.marks == other.marks and (self.bases == other.bases)

    def inferGlyphClasses(self):
        result = {glyph: 1 for glyph in self.bases}
        result.update({glyph: 3 for glyph in self.marks})
        return result

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the mark-to-base
            positioning lookup.
        """
        markClasses = self.buildMarkClasses_(self.marks)
        marks = {}
        for mark, (mc, anchor) in self.marks.items():
            if mc not in markClasses:
                raise ValueError('Mark class %s not found for mark glyph %s' % (mc, mark))
            marks[mark] = (markClasses[mc], anchor)
        bases = {}
        for glyph, anchors in self.bases.items():
            bases[glyph] = {}
            for mc, anchor in anchors.items():
                if mc not in markClasses:
                    raise ValueError('Mark class %s not found for base glyph %s' % (mc, glyph))
                bases[glyph][markClasses[mc]] = anchor
        subtables = buildMarkBasePos(marks, bases, self.glyphMap)
        return self.buildLookup_(subtables)