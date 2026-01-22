from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
def Object_collect_device_varidxes(self, varidxes):
    adder = partial(_Device_recordVarIdx, s=varidxes)
    _visit(self, adder)