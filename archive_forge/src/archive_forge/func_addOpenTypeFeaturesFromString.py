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
def addOpenTypeFeaturesFromString(font, features, filename=None, tables=None, debug=False):
    """Add features from a string to a font. Note that this replaces any
    features currently present.

    Args:
        font (feaLib.ttLib.TTFont): The font object.
        features: A string containing feature code.
        filename: The directory containing ``filename`` is used as the root of
            relative ``include()`` paths; if ``None`` is provided, the current
            directory is assumed.
        tables: If passed, restrict the set of affected tables to those in the
            list.
        debug: Whether to add source debugging information to the font in the
            ``Debg`` table

    """
    featurefile = StringIO(tostr(features))
    if filename:
        featurefile.name = filename
    addOpenTypeFeatures(font, featurefile, tables=tables, debug=debug)