from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
def _add_VarData(self):
    regionMap = self._regionMap
    regionList = self._regionList
    regions = self._supports
    regionIndices = []
    for region in regions:
        key = _getLocationKey(region)
        idx = regionMap.get(key)
        if idx is None:
            varRegion = buildVarRegion(region, self._axisTags)
            idx = regionMap[key] = len(regionList.Region)
            regionList.Region.append(varRegion)
        regionIndices.append(idx)
    key = tuple(regionIndices)
    varDataIdx = self._varDataIndices.get(key)
    if varDataIdx is not None:
        self._outer = varDataIdx
        self._data = self._store.VarData[varDataIdx]
        self._cache = self._varDataCaches[key]
        if len(self._data.Item) == 65535:
            varDataIdx = None
    if varDataIdx is None:
        self._data = buildVarData(regionIndices, [], optimize=False)
        self._outer = len(self._store.VarData)
        self._store.VarData.append(self._data)
        self._varDataIndices[key] = self._outer
        if key not in self._varDataCaches:
            self._varDataCaches[key] = {}
        self._cache = self._varDataCaches[key]