import sys
from rdkit import Chem
from rdkit.Chem.rdfragcatalog import *
def BuildAdjacencyList(catalog, bits, limitInclusion=1, orderLevels=0):
    adjs = {}
    levels = {}
    bitIds = [bit.id for bit in bits]
    for bitId in bitIds:
        entry = catalog.GetBitEntryId(bitId)
        tmp = []
        order = catalog.GetEntryOrder(entry)
        s = levels.get(order, set())
        s.add(bitId)
        levels[order] = s
        for down in catalog.GetEntryDownIds(entry):
            id = catalog.GetEntryBitId(down)
            if not limitInclusion or id in bitIds:
                tmp.append(id)
                order = catalog.GetEntryOrder(down)
                s = levels.get(order, set())
                s.add(id)
                levels[order] = s
        adjs[bitId] = tmp
    if orderLevels:
        for order in levels.keys():
            ids = levels[order]
            counts = [len(adjs[id]) for id in ids]
            countOrder = argsort(counts)
            l = [ids[x] for x in countOrder]
            l.reverse()
            levels[order] = l
    return (adjs, levels)