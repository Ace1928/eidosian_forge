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
def _PairPosFormat2_align_matrices(self, lst, font, transparent=False):
    matrices = [l.Class1Record for l in lst]
    self.ClassDef1, classes = _ClassDef_merge_classify([l.ClassDef1 for l in lst], [l.Coverage.glyphs for l in lst])
    self.Class1Count = len(classes)
    new_matrices = []
    for l, matrix in zip(lst, matrices):
        nullRow = None
        coverage = set(l.Coverage.glyphs)
        classDef1 = l.ClassDef1.classDefs
        class1Records = []
        for classSet in classes:
            exemplarGlyph = next(iter(classSet))
            if exemplarGlyph not in coverage:
                nullRow = None
                if nullRow is None:
                    nullRow = ot.Class1Record()
                    class2records = nullRow.Class2Record = []
                    for _ in range(l.Class2Count):
                        if transparent:
                            rec2 = None
                        else:
                            rec2 = ot.Class2Record()
                            rec2.Value1 = otBase.ValueRecord(self.ValueFormat1) if self.ValueFormat1 else None
                            rec2.Value2 = otBase.ValueRecord(self.ValueFormat2) if self.ValueFormat2 else None
                        class2records.append(rec2)
                rec1 = nullRow
            else:
                klass = classDef1.get(exemplarGlyph, 0)
                rec1 = matrix[klass]
            class1Records.append(rec1)
        new_matrices.append(class1Records)
    matrices = new_matrices
    del new_matrices
    self.ClassDef2, classes = _ClassDef_merge_classify([l.ClassDef2 for l in lst])
    self.Class2Count = len(classes)
    new_matrices = []
    for l, matrix in zip(lst, matrices):
        classDef2 = l.ClassDef2.classDefs
        class1Records = []
        for rec1old in matrix:
            oldClass2Records = rec1old.Class2Record
            rec1new = ot.Class1Record()
            class2Records = rec1new.Class2Record = []
            for classSet in classes:
                if not classSet:
                    rec2 = oldClass2Records[0]
                else:
                    exemplarGlyph = next(iter(classSet))
                    klass = classDef2.get(exemplarGlyph, 0)
                    rec2 = oldClass2Records[klass]
                class2Records.append(copy.deepcopy(rec2))
            class1Records.append(rec1new)
        new_matrices.append(class1Records)
    matrices = new_matrices
    del new_matrices
    return matrices