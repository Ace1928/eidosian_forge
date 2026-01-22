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
def _ClassDef_invert(self, allGlyphs=None):
    if isinstance(self, dict):
        classDefs = self
    else:
        classDefs = self.classDefs if self and self.classDefs else {}
    m = max(classDefs.values()) if classDefs else 0
    ret = []
    for _ in range(m + 1):
        ret.append(set())
    for k, v in classDefs.items():
        ret[v].add(k)
    if allGlyphs is None:
        ret[0] = None
    else:
        ret[0] = class0 = set(allGlyphs)
        for s in ret[1:]:
            s.intersection_update(class0)
            class0.difference_update(s)
    return ret