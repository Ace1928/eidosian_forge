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
def add_pos(self, location, glyph, otValueRecord):
    """Add a single positioning rule.

        Args:
            location: A string or tuple representing the location in the
                original source which produced this lookup.
            glyph: A glyph name.
            otValueRection: A ``otTables.ValueRecord`` used to position the
                glyph.
        """
    if not self.can_add(glyph, otValueRecord):
        otherLoc = self.locations[glyph]
        raise OpenTypeLibError('Already defined different position for glyph "%s" at %s' % (glyph, otherLoc), location)
    if otValueRecord:
        self.mapping[glyph] = otValueRecord
    self.locations[glyph] = location