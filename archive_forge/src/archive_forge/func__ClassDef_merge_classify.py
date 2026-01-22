import os
import copy
import enum
from operator import ior
import logging
from fontTools.colorLib.builder import MAX_PAINT_COLR_LAYER_COUNT, LayerReuseCache
from fontTools.misc import classifyTools
from fontTools.misc.roundTools import otRound
from fontTools.misc.treeTools import build_n_ary_tree
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables import otBase as otBase
from fontTools.ttLib.tables.otConverters import BaseFixedValue
from fontTools.ttLib.tables.otTraverse import dfs_base_table
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.varLib import builder, models, varStore
from fontTools.varLib.models import nonNone, allNone, allEqual, allEqualTo, subList
from fontTools.varLib.varStore import VarStoreInstancer
from functools import reduce
from fontTools.otlLib.builder import buildSinglePos
from fontTools.otlLib.optimize.gpos import (
from .errors import (
def _ClassDef_merge_classify(lst, allGlyphses=None):
    self = ot.ClassDef()
    self.classDefs = classDefs = {}
    allGlyphsesWasNone = allGlyphses is None
    if allGlyphsesWasNone:
        allGlyphses = [None] * len(lst)
    classifier = classifyTools.Classifier()
    for classDef, allGlyphs in zip(lst, allGlyphses):
        sets = _ClassDef_invert(classDef, allGlyphs)
        if allGlyphs is None:
            sets = sets[1:]
        classifier.update(sets)
    classes = classifier.getClasses()
    if allGlyphsesWasNone:
        classes.insert(0, set())
    for i, classSet in enumerate(classes):
        if i == 0:
            continue
        for g in classSet:
            classDefs[g] = i
    return (self, classes)