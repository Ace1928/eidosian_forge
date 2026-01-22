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
class SingleSubstBuilder(LookupBuilder):
    """Builds a Single Substitution (GSUB1) lookup.

    Users are expected to manually add substitutions to the ``mapping``
    attribute after the object has been initialized, e.g.::

        # sub x by y;
        builder.mapping["x"] = "y"

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        mapping: A dictionary mapping a single glyph name to another glyph name.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, 'GSUB', 1)
        self.mapping = OrderedDict()

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.mapping == other.mapping

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the multiple
            substitution lookup.
        """
        subtables = self.build_subst_subtables(self.mapping, buildSingleSubstSubtable)
        return self.buildLookup_(subtables)

    def getAlternateGlyphs(self):
        return {glyph: [repl] for glyph, repl in self.mapping.items()}

    def add_subtable_break(self, location):
        self.mapping[self.SUBTABLE_BREAK_, location] = self.SUBTABLE_BREAK_