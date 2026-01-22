import itertools
def GetIndexCombinations(nItems, nSlots, slot=0, lastItemVal=0):
    """ Generates all combinations of nItems in nSlots without including
      duplicates

    **Arguments**

      - nItems: the number of items to distribute

      - nSlots: the number of slots in which to distribute them

      - slot: used in recursion

      - lastItemVal: used in recursion

    **Returns**

      a list of lists

    """
    global _indexCombinations
    if not slot and (nItems, nSlots) in _indexCombinations:
        res = _indexCombinations[nItems, nSlots]
    elif slot >= nSlots:
        res = []
    elif slot == nSlots - 1:
        res = [[x] for x in range(lastItemVal, nItems)]
    else:
        res = []
        for x in range(lastItemVal, nItems):
            tmp = GetIndexCombinations(nItems, nSlots, slot + 1, x)
            for entry in tmp:
                res.append([x] + entry)
        if not slot:
            _indexCombinations[nItems, nSlots] = res
    return res