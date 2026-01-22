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
class MarkLigPosBuilder(LookupBuilder):
    """Builds a Mark-To-Ligature Positioning (GPOS5) lookup.

    Users are expected to manually add marks and bases to the ``marks``
    and ``ligatures`` attributes after the object has been initialized, e.g.::

        builder.marks["acute"]   = (0, a1)
        builder.marks["grave"]   = (0, a1)
        builder.marks["cedilla"] = (1, a2)
        builder.ligatures["f_i"] = [
            { 0: a3, 1: a5 }, # f
            { 0: a4, 1: a5 }  # i
        ]

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        marks: An dictionary mapping a glyph name to a two-element
            tuple containing a mark class ID and ``otTables.Anchor`` object.
        ligatures: An dictionary mapping a glyph name to an array with one
            element for each ligature component. Each array element should be
            a dictionary mapping mark class IDs to ``otTables.Anchor`` objects.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, 'GPOS', 5)
        self.marks = {}
        self.ligatures = {}

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.marks == other.marks and (self.ligatures == other.ligatures)

    def inferGlyphClasses(self):
        result = {glyph: 2 for glyph in self.ligatures}
        result.update({glyph: 3 for glyph in self.marks})
        return result

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the mark-to-ligature
            positioning lookup.
        """
        markClasses = self.buildMarkClasses_(self.marks)
        marks = {mark: (markClasses[mc], anchor) for mark, (mc, anchor) in self.marks.items()}
        ligs = {}
        for lig, components in self.ligatures.items():
            ligs[lig] = []
            for c in components:
                ligs[lig].append({markClasses[mc]: a for mc, a in c.items()})
        subtables = buildMarkLigPos(marks, ligs, self.glyphMap)
        return self.buildLookup_(subtables)