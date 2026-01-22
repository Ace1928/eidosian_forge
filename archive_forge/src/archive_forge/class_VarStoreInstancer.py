from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
class VarStoreInstancer(object):

    def __init__(self, varstore, fvar_axes, location={}):
        self.fvar_axes = fvar_axes
        assert varstore is None or varstore.Format == 1
        self._varData = varstore.VarData if varstore else []
        self._regions = varstore.VarRegionList.Region if varstore else []
        self.setLocation(location)

    def setLocation(self, location):
        self.location = dict(location)
        self._clearCaches()

    def _clearCaches(self):
        self._scalars = {}

    def _getScalar(self, regionIdx):
        scalar = self._scalars.get(regionIdx)
        if scalar is None:
            support = self._regions[regionIdx].get_support(self.fvar_axes)
            scalar = supportScalar(self.location, support)
            self._scalars[regionIdx] = scalar
        return scalar

    @staticmethod
    def interpolateFromDeltasAndScalars(deltas, scalars):
        delta = 0.0
        for d, s in zip(deltas, scalars):
            if not s:
                continue
            delta += d * s
        return delta

    def __getitem__(self, varidx):
        major, minor = (varidx >> 16, varidx & 65535)
        if varidx == NO_VARIATION_INDEX:
            return 0.0
        varData = self._varData
        scalars = [self._getScalar(ri) for ri in varData[major].VarRegionIndex]
        deltas = varData[major].Item[minor]
        return self.interpolateFromDeltasAndScalars(deltas, scalars)

    def interpolateFromDeltas(self, varDataIndex, deltas):
        varData = self._varData
        scalars = [self._getScalar(ri) for ri in varData[varDataIndex].VarRegionIndex]
        return self.interpolateFromDeltasAndScalars(deltas, scalars)