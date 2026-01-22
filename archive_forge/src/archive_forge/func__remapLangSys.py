from fontTools.misc.dictTools import hashdict
from fontTools.misc.intTools import bit_count
from fontTools.ttLib import newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.ttVisitor import TTVisitor
from fontTools.otlLib.builder import buildLookup, buildSingleSubstSubtable
from collections import OrderedDict
from .errors import VarLibError, VarLibValidationError
def _remapLangSys(langSys, featureRemap):
    if langSys.ReqFeatureIndex != 65535:
        langSys.ReqFeatureIndex = featureRemap[langSys.ReqFeatureIndex]
    langSys.FeatureIndex = [featureRemap[index] for index in langSys.FeatureIndex]