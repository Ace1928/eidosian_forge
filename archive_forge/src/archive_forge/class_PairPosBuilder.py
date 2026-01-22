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
class PairPosBuilder(LookupBuilder):
    """Builds a Pair Positioning (GPOS2) lookup.

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        pairs: An array of class-based pair positioning tuples. Usually
            manipulated with the :meth:`addClassPair` method below.
        glyphPairs: A dictionary mapping a tuple of glyph names to a tuple
            of ``otTables.ValueRecord`` objects. Usually manipulated with the
            :meth:`addGlyphPair` method below.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, 'GPOS', 2)
        self.pairs = []
        self.glyphPairs = {}
        self.locations = {}

    def addClassPair(self, location, glyphclass1, value1, glyphclass2, value2):
        """Add a class pair positioning rule to the current lookup.

        Args:
            location: A string or tuple representing the location in the
                original source which produced this rule. Unused.
            glyphclass1: A set of glyph names for the "left" glyph in the pair.
            value1: A ``otTables.ValueRecord`` for positioning the left glyph.
            glyphclass2: A set of glyph names for the "right" glyph in the pair.
            value2: A ``otTables.ValueRecord`` for positioning the right glyph.
        """
        self.pairs.append((glyphclass1, value1, glyphclass2, value2))

    def addGlyphPair(self, location, glyph1, value1, glyph2, value2):
        """Add a glyph pair positioning rule to the current lookup.

        Args:
            location: A string or tuple representing the location in the
                original source which produced this rule.
            glyph1: A glyph name for the "left" glyph in the pair.
            value1: A ``otTables.ValueRecord`` for positioning the left glyph.
            glyph2: A glyph name for the "right" glyph in the pair.
            value2: A ``otTables.ValueRecord`` for positioning the right glyph.
        """
        key = (glyph1, glyph2)
        oldValue = self.glyphPairs.get(key, None)
        if oldValue is not None:
            otherLoc = self.locations[key]
            log.debug('Already defined position for pair %s %s at %s; choosing the first value', glyph1, glyph2, otherLoc)
        else:
            self.glyphPairs[key] = (value1, value2)
            self.locations[key] = location

    def add_subtable_break(self, location):
        self.pairs.append((self.SUBTABLE_BREAK_, self.SUBTABLE_BREAK_, self.SUBTABLE_BREAK_, self.SUBTABLE_BREAK_))

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.glyphPairs == other.glyphPairs and (self.pairs == other.pairs)

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the pair positioning
            lookup.
        """
        builders = {}
        builder = ClassPairPosSubtableBuilder(self)
        for glyphclass1, value1, glyphclass2, value2 in self.pairs:
            if glyphclass1 is self.SUBTABLE_BREAK_:
                builder.addSubtableBreak()
                continue
            builder.addPair(glyphclass1, value1, glyphclass2, value2)
        subtables = []
        if self.glyphPairs:
            subtables.extend(buildPairPosGlyphs(self.glyphPairs, self.glyphMap))
        subtables.extend(builder.subtables())
        lookup = self.buildLookup_(subtables)
        level = self.font.cfg.get('fontTools.otlLib.optimize.gpos:COMPRESSION_LEVEL', default=_compression_level_from_env())
        if level != 0:
            log.info('Compacting GPOS...')
            compact_lookup(self.font, level, lookup)
        return lookup