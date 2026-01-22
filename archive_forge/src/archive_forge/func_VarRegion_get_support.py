from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
def VarRegion_get_support(self, fvar_axes):
    return {fvar_axes[i].axisTag: (reg.StartCoord, reg.PeakCoord, reg.EndCoord) for i, reg in enumerate(self.VarRegionAxis) if reg.PeakCoord != 0}