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
def _buildMathGlyphConstruction(glyphMap, variants, assemblies):
    glyphs = set()
    if variants:
        glyphs.update(variants.keys())
    if assemblies:
        glyphs.update(assemblies.keys())
    coverage = buildCoverage(glyphs, glyphMap)
    constructions = []
    for glyphName in coverage.glyphs:
        construction = ot.MathGlyphConstruction()
        construction.populateDefaults()
        if variants and glyphName in variants:
            construction.VariantCount = len(variants[glyphName])
            construction.MathGlyphVariantRecord = []
            for variantName, advance in variants[glyphName]:
                record = ot.MathGlyphVariantRecord()
                record.VariantGlyph = variantName
                record.AdvanceMeasurement = otRound(advance)
                construction.MathGlyphVariantRecord.append(record)
        if assemblies and glyphName in assemblies:
            parts, ic = assemblies[glyphName]
            construction.GlyphAssembly = ot.GlyphAssembly()
            construction.GlyphAssembly.ItalicsCorrection = _mathValueRecord(ic)
            construction.GlyphAssembly.PartCount = len(parts)
            construction.GlyphAssembly.PartRecords = []
            for part in parts:
                part_name, flags, start, end, advance = part
                record = ot.GlyphPartRecord()
                record.glyph = part_name
                record.PartFlags = int(flags)
                record.StartConnectorLength = otRound(start)
                record.EndConnectorLength = otRound(end)
                record.FullAdvance = otRound(advance)
                construction.GlyphAssembly.PartRecords.append(record)
        constructions.append(construction)
    return (coverage, constructions)