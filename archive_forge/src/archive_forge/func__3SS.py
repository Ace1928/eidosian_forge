import numpy as np
import os
import time
from ..utils import *
def _3SS(imgP, imgI, mbSize, p):
    h, w = imgP.shape
    vectors = np.zeros((np.int(h / mbSize), np.int(w / mbSize), 2))
    costs = np.ones((3, 3), dtype=np.float32) * 65537
    computations = 0
    L = np.floor(np.log2(p + 1))
    stepMax = np.int(2 ** (L - 1))
    for i in range(0, h - mbSize + 1, mbSize):
        for j in range(0, w - mbSize + 1, mbSize):
            x = j
            y = i
            costs[1, 1] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[i:i + mbSize, j:j + mbSize])
            computations += 1
            stepSize = stepMax
            while stepSize >= 1:
                for m in range(-stepSize, stepSize + 1, stepSize):
                    for n in range(-stepSize, stepSize + 1, stepSize):
                        refBlkVer = y + m
                        refBlkHor = x + n
                        if refBlkVer < 0 or refBlkVer + mbSize > h or refBlkHor < 0 or (refBlkHor + mbSize > w):
                            continue
                        costRow = np.int(m / stepSize) + 1
                        costCol = np.int(n / stepSize) + 1
                        if costRow == 1 and costCol == 1:
                            continue
                        costs[costRow, costCol] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                        computations = computations + 1
                dx, dy, mi = _minCost(costs)
                x += (dx - 1) * stepSize
                y += (dy - 1) * stepSize
                stepSize = np.int(stepSize / 2)
                costs[1, 1] = costs[dy, dx]
            vectors[np.int(i / mbSize), np.int(j / mbSize), :] = [y - i, x - j]
            costs[:, :] = 65537
    return (vectors, computations / (h * w / mbSize ** 2))