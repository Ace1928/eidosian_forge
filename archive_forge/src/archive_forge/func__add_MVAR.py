from typing import List
from fontTools.misc.vector import Vector
from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.fixedTools import floatToFixed as fl2fi
from fontTools.misc.textTools import Tag, tostr
from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables._f_v_a_r import Axis, NamedInstance
from fontTools.ttLib.tables._g_l_y_f import GlyphCoordinates, dropImpliedOnCurvePoints
from fontTools.ttLib.tables.ttProgram import Program
from fontTools.ttLib.tables.TupleVariation import TupleVariation
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import OTTableWriter
from fontTools.varLib import builder, models, varStore
from fontTools.varLib.merger import VariationMerger, COLRVariationMerger
from fontTools.varLib.mvar import MVAR_ENTRIES
from fontTools.varLib.iup import iup_delta_optimize
from fontTools.varLib.featureVars import addFeatureVariations
from fontTools.designspaceLib import DesignSpaceDocument, InstanceDescriptor
from fontTools.designspaceLib.split import splitInterpolable, splitVariableFonts
from fontTools.varLib.stat import buildVFStatTable
from fontTools.colorLib.builder import buildColrV1
from fontTools.colorLib.unbuilder import unbuildColrV1
from functools import partial
from collections import OrderedDict, defaultdict, namedtuple
import os.path
import logging
from copy import deepcopy
from pprint import pformat
from re import fullmatch
from .errors import VarLibError, VarLibValidationError
def _add_MVAR(font, masterModel, master_ttfs, axisTags):
    log.info('Generating MVAR')
    store_builder = varStore.OnlineVarStoreBuilder(axisTags)
    records = []
    lastTableTag = None
    fontTable = None
    tables = None
    specialTags = {'unds': -32768, 'undo': -32768}
    for tag, (tableTag, itemName) in sorted(MVAR_ENTRIES.items(), key=lambda kv: kv[1]):
        if tableTag != lastTableTag:
            tables = fontTable = None
            if tableTag in font:
                fontTable = font[tableTag]
                tables = []
                for master in master_ttfs:
                    if tableTag not in master or (tag in specialTags and getattr(master[tableTag], itemName) == specialTags[tag]):
                        tables.append(None)
                    else:
                        tables.append(master[tableTag])
                model, tables = masterModel.getSubModel(tables)
                store_builder.setModel(model)
            lastTableTag = tableTag
        if tables is None:
            continue
        master_values = [getattr(table, itemName) for table in tables]
        if models.allEqual(master_values):
            base, varIdx = (master_values[0], None)
        else:
            base, varIdx = store_builder.storeMasters(master_values)
        setattr(fontTable, itemName, base)
        if varIdx is None:
            continue
        log.info('\t%s: %s.%s\t%s', tag, tableTag, itemName, master_values)
        rec = ot.MetricsValueRecord()
        rec.ValueTag = tag
        rec.VarIdx = varIdx
        records.append(rec)
    assert 'MVAR' not in font
    if records:
        store = store_builder.finish()
        mapping = store.optimize()
        for rec in records:
            rec.VarIdx = mapping[rec.VarIdx]
        MVAR = font['MVAR'] = newTable('MVAR')
        mvar = MVAR.table = ot.MVAR()
        mvar.Version = 65536
        mvar.Reserved = 0
        mvar.VarStore = store
        mvar.ValueRecordSize = 8
        mvar.ValueRecordCount = len(records)
        mvar.ValueRecord = sorted(records, key=lambda r: r.ValueTag)