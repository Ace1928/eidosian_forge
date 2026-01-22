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
def _buildMathVariants(glyphMap, minConnectorOverlap, vertGlyphVariants, horizGlyphVariants, vertGlyphAssembly, horizGlyphAssembly):
    if not any([vertGlyphVariants, horizGlyphVariants, vertGlyphAssembly, horizGlyphAssembly]):
        return None
    variants = ot.MathVariants()
    variants.populateDefaults()
    variants.MinConnectorOverlap = minConnectorOverlap
    if vertGlyphVariants or vertGlyphAssembly:
        variants.VertGlyphCoverage, variants.VertGlyphConstruction = _buildMathGlyphConstruction(glyphMap, vertGlyphVariants, vertGlyphAssembly)
    if horizGlyphVariants or horizGlyphAssembly:
        variants.HorizGlyphCoverage, variants.HorizGlyphConstruction = _buildMathGlyphConstruction(glyphMap, horizGlyphVariants, horizGlyphAssembly)
    return variants