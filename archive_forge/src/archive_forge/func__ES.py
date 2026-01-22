import numpy as np
import os
import time
from ..utils import *
def _ES(imgP, imgI, mbSize, p):
    h, w = imgP.shape
    vectors = np.zeros((np.int(h / mbSize), np.int(w / mbSize), 2), dtype=np.float32)
    costs = np.ones((2 * p + 1, 2 * p + 1), dtype=np.float32) * 65537
    for i in range(0, h - mbSize + 1, mbSize):
        for j in range(0, w - mbSize + 1, mbSize):
            if j + p + mbSize >= w or j - p < 0 or i - p < 0 or (i + p + mbSize >= h):
                for m in range(-p, p + 1):
                    for n in range(-p, p + 1):
                        refBlkVer = i + m
                        refBlkHor = j + n
                        if refBlkVer < 0 or refBlkVer + mbSize > h or refBlkHor < 0 or (refBlkHor + mbSize > w):
                            continue
                        costs[m + p, n + p] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
            else:
                for m in range(-p, p + 1):
                    for n in range(-p, p + 1):
                        refBlkVer = i + m
                        refBlkHor = j + n
                        costs[m + p, n + p] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
            dx, dy, mi = _minCost(costs)
            vectors[np.int(i / mbSize), np.int(j / mbSize), :] = [dy - p, dx - p]
            costs[:, :] = 65537
    return vectors