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
class MarkMarkPosBuilder(LookupBuilder):
    """Builds a Mark-To-Mark Positioning (GPOS6) lookup.

    Users are expected to manually add marks and bases to the ``marks``
    and ``baseMarks`` attributes after the object has been initialized, e.g.::

        builder.marks["acute"]     = (0, a1)
        builder.marks["grave"]     = (0, a1)
        builder.marks["cedilla"]   = (1, a2)
        builder.baseMarks["acute"] = {0: a3}

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        marks: An dictionary mapping a glyph name to a two-element
            tuple containing a mark class ID and ``otTables.Anchor`` object.
        baseMarks: An dictionary mapping a glyph name to a dictionary
            containing one item: a mark class ID and a ``otTables.Anchor`` object.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, 'GPOS', 6)
        self.marks = {}
        self.baseMarks = {}

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.marks == other.marks and (self.baseMarks == other.baseMarks)

    def inferGlyphClasses(self):
        result = {glyph: 3 for glyph in self.baseMarks}
        result.update({glyph: 3 for glyph in self.marks})
        return result

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the mark-to-mark
            positioning lookup.
        """
        markClasses = self.buildMarkClasses_(self.marks)
        markClassList = sorted(markClasses.keys(), key=markClasses.get)
        marks = {mark: (markClasses[mc], anchor) for mark, (mc, anchor) in self.marks.items()}
        st = ot.MarkMarkPos()
        st.Format = 1
        st.ClassCount = len(markClasses)
        st.Mark1Coverage = buildCoverage(marks, self.glyphMap)
        st.Mark2Coverage = buildCoverage(self.baseMarks, self.glyphMap)
        st.Mark1Array = buildMarkArray(marks, self.glyphMap)
        st.Mark2Array = ot.Mark2Array()
        st.Mark2Array.Mark2Count = len(st.Mark2Coverage.glyphs)
        st.Mark2Array.Mark2Record = []
        for base in st.Mark2Coverage.glyphs:
            anchors = [self.baseMarks[base].get(mc) for mc in markClassList]
            st.Mark2Array.Mark2Record.append(buildMark2Record(anchors))
        return self.buildLookup_([st])