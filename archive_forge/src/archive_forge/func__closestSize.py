from rdkit.sping.pid import *
import math
import os
def _closestSize(size):
    supported = [8, 10, 12, 14, 18, 24]
    if size in supported:
        return size
    best = supported[0]
    bestdist = abs(size - best)
    for trial in supported[1:]:
        dist = abs(size - trial)
        if dist < bestdist:
            best = trial
            bestdist = dist
    return best