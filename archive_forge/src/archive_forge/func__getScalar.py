from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
def _getScalar(self, regionIdx):
    scalar = self._scalars.get(regionIdx)
    if scalar is None:
        support = self._regions[regionIdx].get_support(self.fvar_axes)
        scalar = supportScalar(self.location, support)
        self._scalars[regionIdx] = scalar
    return scalar