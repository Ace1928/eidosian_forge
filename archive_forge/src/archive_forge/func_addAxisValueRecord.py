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
def addAxisValueRecord(self, axisValueRecord, location):
    if 'AxisValueRecords' not in self.stat_:
        self.stat_['AxisValueRecords'] = []
    for record_ in self.stat_['AxisValueRecords']:
        if {n.asFea() for n in record_.names} == {n.asFea() for n in axisValueRecord.names} and {n.asFea() for n in record_.locations} == {n.asFea() for n in axisValueRecord.locations} and (record_.flags == axisValueRecord.flags):
            raise FeatureLibError('An AxisValueRecord with these values is already defined.', location)
    self.stat_['AxisValueRecords'].append(axisValueRecord)