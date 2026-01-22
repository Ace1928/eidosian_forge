from fontTools import config
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables
from fontTools.ttLib.tables.otBase import USE_HARFBUZZ_REPACKER
from fontTools.otlLib.maxContextCalc import maxCtxFont
from fontTools.pens.basePen import NullPen
from fontTools.misc.loggingTools import Timer
from fontTools.misc.cliTools import makeOutputFileName
from fontTools.subset.util import _add_method, _uniq_sort
from fontTools.subset.cff import *
from fontTools.subset.svg import *
from fontTools.varLib import varStore  # for subset_varidxes
from fontTools.ttLib.tables._n_a_m_e import NameRecordVisitor
import sys
import struct
import array
import logging
from collections import Counter, defaultdict
from functools import reduce
from types import MethodType
def _pruneGDEF(font):
    if 'GDEF' not in font:
        return
    gdef = font['GDEF']
    table = gdef.table
    if not hasattr(table, 'VarStore'):
        return
    store = table.VarStore
    usedVarIdxes = set()
    table.collect_device_varidxes(usedVarIdxes)
    if 'GPOS' in font:
        font['GPOS'].table.collect_device_varidxes(usedVarIdxes)
    varidx_map = store.subset_varidxes(usedVarIdxes)
    table.remap_device_varidxes(varidx_map)
    if 'GPOS' in font:
        font['GPOS'].table.remap_device_varidxes(varidx_map)