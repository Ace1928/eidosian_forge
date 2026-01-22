from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
def VarStore_subset_varidxes(self, varIdxes, optimize=True, retainFirstMap=False, advIdxes=set()):
    used = {}
    for varIdx in varIdxes:
        if varIdx == NO_VARIATION_INDEX:
            continue
        major = varIdx >> 16
        minor = varIdx & 65535
        d = used.get(major)
        if d is None:
            d = used[major] = set()
        d.add(minor)
    del varIdxes
    varData = self.VarData
    newVarData = []
    varDataMap = {NO_VARIATION_INDEX: NO_VARIATION_INDEX}
    for major, data in enumerate(varData):
        usedMinors = used.get(major)
        if usedMinors is None:
            continue
        newMajor = len(newVarData)
        newVarData.append(data)
        items = data.Item
        newItems = []
        if major == 0 and retainFirstMap:
            for minor in range(len(items)):
                newItems.append(items[minor] if minor in usedMinors else [0] * len(items[minor]))
                varDataMap[minor] = minor
        else:
            if major == 0:
                minors = sorted(advIdxes) + sorted(usedMinors - advIdxes)
            else:
                minors = sorted(usedMinors)
            for minor in minors:
                newMinor = len(newItems)
                newItems.append(items[minor])
                varDataMap[(major << 16) + minor] = (newMajor << 16) + newMinor
        data.Item = newItems
        data.ItemCount = len(data.Item)
        data.calculateNumShorts(optimize=optimize)
    self.VarData = newVarData
    self.VarDataCount = len(self.VarData)
    self.prune_regions()
    return varDataMap