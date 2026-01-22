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
def add_single_subst_chained_(self, location, prefix, suffix, mapping):
    if not mapping or not all(prefix) or (not all(suffix)):
        raise FeatureLibError('Empty glyph class in contextual substitution', location)
    chain = self.get_lookup_(location, ChainContextSubstBuilder)
    sub = chain.find_chainable_single_subst(mapping)
    if sub is None:
        sub = self.get_chained_lookup_(location, SingleSubstBuilder)
    sub.mapping.update(mapping)
    chain.rules.append(ChainContextualRule(prefix, [list(mapping.keys())], suffix, [sub]))