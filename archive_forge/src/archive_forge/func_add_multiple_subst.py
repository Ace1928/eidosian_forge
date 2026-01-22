from fontTools.misc import sstruct
from fontTools.misc.textTools import Tag, tostr, binary2num, safeEval
from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lookupDebugInfo import (
from fontTools.feaLib.parser import Parser
from fontTools.feaLib.ast import FeatureFile
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.otlLib import builder as otl
from fontTools.otlLib.maxContextCalc import maxCtxFont
from fontTools.ttLib import newTable, getTableModule
from fontTools.ttLib.tables import otBase, otTables
from fontTools.otlLib.builder import (
from fontTools.otlLib.error import OpenTypeLibError
from fontTools.varLib.varStore import OnlineVarStoreBuilder
from fontTools.varLib.builder import buildVarDevTable
from fontTools.varLib.featureVars import addFeatureVariationsRaw
from fontTools.varLib.models import normalizeValue, piecewiseLinearMap
from collections import defaultdict
import copy
import itertools
from io import StringIO
import logging
import warnings
import os
def add_multiple_subst(self, location, prefix, glyph, suffix, replacements, forceChain=False):
    if prefix or suffix or forceChain:
        chain = self.get_lookup_(location, ChainContextSubstBuilder)
        sub = self.get_chained_lookup_(location, MultipleSubstBuilder)
        sub.mapping[glyph] = replacements
        chain.rules.append(ChainContextualRule(prefix, [{glyph}], suffix, [sub]))
        return
    lookup = self.get_lookup_(location, MultipleSubstBuilder)
    if glyph in lookup.mapping:
        if replacements == lookup.mapping[glyph]:
            log.info('Removing duplicate multiple substitution from glyph "%s" to %s%s', glyph, replacements, f' at {location}' if location else '')
        else:
            raise FeatureLibError('Already defined substitution for glyph "%s"' % glyph, location)
    lookup.mapping[glyph] = replacements