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
class LigatureSubstBuilder(LookupBuilder):
    """Builds a Ligature Substitution (GSUB4) lookup.

    Users are expected to manually add ligatures to the ``ligatures``
    attribute after the object has been initialized, e.g.::

        # sub f i by f_i;
        builder.ligatures[("f","f","i")] = "f_f_i"

    Attributes:
        font (``fontTools.TTLib.TTFont``): A font object.
        location: A string or tuple representing the location in the original
            source which produced this lookup.
        ligatures: An ordered dictionary mapping a tuple of glyph names to the
            ligature glyphname.
        lookupflag (int): The lookup's flag
        markFilterSet: Either ``None`` if no mark filtering set is used, or
            an integer representing the filtering set to be used for this
            lookup. If a mark filtering set is provided,
            `LOOKUP_FLAG_USE_MARK_FILTERING_SET` will be set on the lookup's
            flags.
    """

    def __init__(self, font, location):
        LookupBuilder.__init__(self, font, location, 'GSUB', 4)
        self.ligatures = OrderedDict()

    def equals(self, other):
        return LookupBuilder.equals(self, other) and self.ligatures == other.ligatures

    def build(self):
        """Build the lookup.

        Returns:
            An ``otTables.Lookup`` object representing the ligature
            substitution lookup.
        """
        subtables = self.build_subst_subtables(self.ligatures, buildLigatureSubstSubtable)
        return self.buildLookup_(subtables)

    def add_subtable_break(self, location):
        self.ligatures[self.SUBTABLE_BREAK_, location] = self.SUBTABLE_BREAK_