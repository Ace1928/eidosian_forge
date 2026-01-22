from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
class OnlineVarStoreBuilder(object):

    def __init__(self, axisTags):
        self._axisTags = axisTags
        self._regionMap = {}
        self._regionList = buildVarRegionList([], axisTags)
        self._store = buildVarStore(self._regionList, [])
        self._data = None
        self._model = None
        self._supports = None
        self._varDataIndices = {}
        self._varDataCaches = {}
        self._cache = {}

    def setModel(self, model):
        self.setSupports(model.supports)
        self._model = model

    def setSupports(self, supports):
        self._model = None
        self._supports = list(supports)
        if not self._supports[0]:
            del self._supports[0]
        self._cache = {}
        self._data = None

    def finish(self, optimize=True):
        self._regionList.RegionCount = len(self._regionList.Region)
        self._store.VarDataCount = len(self._store.VarData)
        for data in self._store.VarData:
            data.ItemCount = len(data.Item)
            data.calculateNumShorts(optimize=optimize)
        return self._store

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

    def storeMasters(self, master_values, *, round=round):
        deltas = self._model.getDeltas(master_values, round=round)
        base = deltas.pop(0)
        return (base, self.storeDeltas(deltas, round=noRound))

    def storeDeltas(self, deltas, *, round=round):
        deltas = [round(d) for d in deltas]
        if len(deltas) == len(self._supports) + 1:
            deltas = tuple(deltas[1:])
        else:
            assert len(deltas) == len(self._supports)
            deltas = tuple(deltas)
        varIdx = self._cache.get(deltas)
        if varIdx is not None:
            return varIdx
        if not self._data:
            self._add_VarData()
        inner = len(self._data.Item)
        if inner == 65535:
            self._add_VarData()
            return self.storeDeltas(deltas)
        self._data.addItem(deltas, round=noRound)
        varIdx = (self._outer << 16) + inner
        self._cache[deltas] = varIdx
        return varIdx