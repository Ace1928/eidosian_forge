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
class ChainContextualRuleset:

    def __init__(self):
        self.rules = []

    def addRule(self, rule):
        self.rules.append(rule)

    @property
    def hasPrefixOrSuffix(self):
        for rule in self.rules:
            if len(rule.prefix) > 0 or len(rule.suffix) > 0:
                return True
        return False

    @property
    def hasAnyGlyphClasses(self):
        for rule in self.rules:
            for coverage in (rule.prefix, rule.glyphs, rule.suffix):
                if any((len(x) > 1 for x in coverage)):
                    return True
        return False

    def format2ClassDefs(self):
        PREFIX, GLYPHS, SUFFIX = (0, 1, 2)
        classDefBuilders = []
        for ix in [PREFIX, GLYPHS, SUFFIX]:
            context = []
            for r in self.rules:
                context.append(r[ix])
            classes = self._classBuilderForContext(context)
            if not classes:
                return None
            classDefBuilders.append(classes)
        return classDefBuilders

    def _classBuilderForContext(self, context):
        classdefbuilder = ClassDefBuilder(useClass0=False)
        for position in context:
            for glyphset in position:
                glyphs = set(glyphset)
                if not classdefbuilder.canAdd(glyphs):
                    return None
                classdefbuilder.add(glyphs)
        return classdefbuilder