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
class ClassPairPosSubtableBuilder(object):
    """Builds class-based Pair Positioning (GPOS2 format 2) subtables.

    Note that this does *not* build a GPOS2 ``otTables.Lookup`` directly,
    but builds a list of ``otTables.PairPos`` subtables. It is used by the
    :class:`PairPosBuilder` below.

    Attributes:
        builder (PairPosBuilder): A pair positioning lookup builder.
    """

    def __init__(self, builder):
        self.builder_ = builder
        self.classDef1_, self.classDef2_ = (None, None)
        self.values_ = {}
        self.forceSubtableBreak_ = False
        self.subtables_ = []

    def addPair(self, gc1, value1, gc2, value2):
        """Add a pair positioning rule.

        Args:
            gc1: A set of glyph names for the "left" glyph
            value1: An ``otTables.ValueRecord`` object for the left glyph's
                positioning.
            gc2: A set of glyph names for the "right" glyph
            value2: An ``otTables.ValueRecord`` object for the right glyph's
                positioning.
        """
        mergeable = not self.forceSubtableBreak_ and self.classDef1_ is not None and self.classDef1_.canAdd(gc1) and (self.classDef2_ is not None) and self.classDef2_.canAdd(gc2)
        if not mergeable:
            self.flush_()
            self.classDef1_ = ClassDefBuilder(useClass0=True)
            self.classDef2_ = ClassDefBuilder(useClass0=False)
            self.values_ = {}
        self.classDef1_.add(gc1)
        self.classDef2_.add(gc2)
        self.values_[gc1, gc2] = (value1, value2)

    def addSubtableBreak(self):
        """Add an explicit subtable break at this point."""
        self.forceSubtableBreak_ = True

    def subtables(self):
        """Return the list of ``otTables.PairPos`` subtables constructed."""
        self.flush_()
        return self.subtables_

    def flush_(self):
        if self.classDef1_ is None or self.classDef2_ is None:
            return
        st = buildPairPosClassesSubtable(self.values_, self.builder_.glyphMap)
        if st.Coverage is None:
            return
        self.subtables_.append(st)
        self.forceSubtableBreak_ = False