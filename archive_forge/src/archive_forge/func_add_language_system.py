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
def add_language_system(self, location, script, language):
    if script == 'DFLT' and language == 'dflt' and self.default_language_systems_:
        raise FeatureLibError('If "languagesystem DFLT dflt" is present, it must be the first of the languagesystem statements', location)
    if script == 'DFLT':
        if self.seen_non_DFLT_script_:
            raise FeatureLibError('languagesystems using the "DFLT" script tag must precede all other languagesystems', location)
    else:
        self.seen_non_DFLT_script_ = True
    if (script, language) in self.default_language_systems_:
        raise FeatureLibError('"languagesystem %s %s" has already been specified' % (script.strip(), language.strip()), location)
    self.default_language_systems_.add((script, language))