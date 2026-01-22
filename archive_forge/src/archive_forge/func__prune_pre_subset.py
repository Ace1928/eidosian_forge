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
def _prune_pre_subset(self, font):
    for tag in self._sort_tables(font):
        if tag.strip() in self.options.drop_tables or (tag.strip() in self.options.hinting_tables and (not self.options.hinting)) or (tag == 'kern' and (not self.options.legacy_kern and 'GPOS' in font)):
            log.info('%s dropped', tag)
            del font[tag]
            continue
        clazz = ttLib.getTableClass(tag)
        if hasattr(clazz, 'prune_pre_subset'):
            with timer("load '%s'" % tag):
                table = font[tag]
            with timer("prune '%s'" % tag):
                retain = table.prune_pre_subset(font, self.options)
            if not retain:
                log.info('%s pruned to empty; dropped', tag)
                del font[tag]
                continue
            else:
                log.info('%s pruned', tag)