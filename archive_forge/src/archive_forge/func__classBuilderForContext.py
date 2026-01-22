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
def _classBuilderForContext(self, context):
    classdefbuilder = ClassDefBuilder(useClass0=False)
    for position in context:
        for glyphset in position:
            glyphs = set(glyphset)
            if not classdefbuilder.canAdd(glyphs):
                return None
            classdefbuilder.add(glyphs)
    return classdefbuilder