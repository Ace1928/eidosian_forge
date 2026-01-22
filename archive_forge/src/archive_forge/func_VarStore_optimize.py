from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
def VarStore_optimize(self, use_NO_VARIATION_INDEX=True, quantization=1):
    """Optimize storage. Returns mapping from old VarIdxes to new ones."""
    n = len(self.VarRegionList.Region)
    zeroes = [0] * n
    front_mapping = {}
    encodings = _EncodingDict()
    for major, data in enumerate(self.VarData):
        regionIndices = data.VarRegionIndex
        for minor, item in enumerate(data.Item):
            row = list(zeroes)
            if quantization == 1:
                for regionIdx, v in zip(regionIndices, item):
                    row[regionIdx] += v
            else:
                for regionIdx, v in zip(regionIndices, item):
                    row[regionIdx] += round(v / quantization) * quantization
            row = tuple(row)
            if use_NO_VARIATION_INDEX and (not any(row)):
                front_mapping[(major << 16) + minor] = None
                continue
            encodings.add_row(row)
            front_mapping[(major << 16) + minor] = row
    todo = sorted(encodings.values(), key=_Encoding.gain_sort_key)
    del encodings
    heap = []
    for i, encoding in enumerate(todo):
        for j in range(i + 1, len(todo)):
            other_encoding = todo[j]
            combining_gain = encoding.gain_from_merging(other_encoding)
            if combining_gain > 0:
                heappush(heap, (-combining_gain, i, j))
    while heap:
        _, i, j = heappop(heap)
        if todo[i] is None or todo[j] is None:
            continue
        encoding, other_encoding = (todo[i], todo[j])
        todo[i], todo[j] = (None, None)
        combined_chars = other_encoding.chars | encoding.chars
        combined_encoding = _Encoding(combined_chars)
        combined_encoding.extend(encoding.items)
        combined_encoding.extend(other_encoding.items)
        for k, enc in enumerate(todo):
            if enc is None:
                continue
            if enc.chars == combined_chars:
                combined_encoding.extend(enc.items)
                todo[k] = None
                continue
            combining_gain = combined_encoding.gain_from_merging(enc)
            if combining_gain > 0:
                heappush(heap, (-combining_gain, k, len(todo)))
        todo.append(combined_encoding)
    encodings = [encoding for encoding in todo if encoding is not None]
    back_mapping = {}
    encodings.sort(key=_Encoding.width_sort_key)
    self.VarData = []
    for encoding in encodings:
        items = sorted(encoding.items)
        while items:
            major = len(self.VarData)
            data = ot.VarData()
            self.VarData.append(data)
            data.VarRegionIndex = range(n)
            data.VarRegionCount = len(data.VarRegionIndex)
            data.Item, items = (items[:65535], items[65535:])
            for minor, item in enumerate(data.Item):
                back_mapping[item] = (major << 16) + minor
    varidx_map = {NO_VARIATION_INDEX: NO_VARIATION_INDEX}
    for k, v in front_mapping.items():
        varidx_map[k] = back_mapping[v] if v is not None else NO_VARIATION_INDEX
    self.VarRegionList.RegionCount = len(self.VarRegionList.Region)
    self.VarDataCount = len(self.VarData)
    for data in self.VarData:
        data.ItemCount = len(data.Item)
        data.optimize()
    self.prune_regions()
    return varidx_map