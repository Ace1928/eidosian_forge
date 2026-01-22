import numpy as np
import os
import time
from ..utils import *
def _subcomp(framedata, motionVect, mbSize):
    M, N, C = framedata.shape
    compImg = np.zeros((M, N, C))
    for i in range(0, M - mbSize + 1, mbSize):
        for j in range(0, N - mbSize + 1, mbSize):
            dy = motionVect[np.int(i / mbSize), np.int(j / mbSize), 0]
            dx = motionVect[np.int(i / mbSize), np.int(j / mbSize), 1]
            refBlkVer = i + dy
            refBlkHor = j + dx
            if not _checkBounded(refBlkHor, refBlkVer, N, M, mbSize):
                continue
            compImg[i:i + mbSize, j:j + mbSize, :] = framedata[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize, :]
    return compImg