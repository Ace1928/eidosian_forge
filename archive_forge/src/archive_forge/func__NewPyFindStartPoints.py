import numpy
from rdkit.ML.InfoTheory import entropy
def _NewPyFindStartPoints(sortVals, sortResults, nData):
    startNext = []
    tol = 1e-08
    blockAct = sortResults[0]
    lastBlockAct = None
    lastDiv = None
    i = 1
    while i < nData:
        while i < nData and sortVals[i] - sortVals[i - 1] <= tol:
            if sortResults[i] != blockAct:
                blockAct = -1
            i += 1
        if lastBlockAct is None:
            lastBlockAct = blockAct
            lastDiv = i
        elif blockAct == -1 or lastBlockAct == -1 or blockAct != lastBlockAct:
            startNext.append(lastDiv)
            lastDiv = i
            lastBlockAct = blockAct
        else:
            lastDiv = i
        if i < nData:
            blockAct = sortResults[i]
        i += 1
    if blockAct != lastBlockAct:
        startNext.append(lastDiv)
    return startNext