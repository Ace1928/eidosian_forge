import numpy as np
import os
import time
from ..utils import *
def _SE3SS(imgP, imgI, mbSize, p):
    h, w = imgP.shape
    vectors = np.zeros((np.int(h / mbSize), np.int(w / mbSize), 2))
    L = np.floor(np.log2(p + 1))
    stepMax = 2 ** (L - 1)
    costs = np.ones(6) * 65537
    computations = 0
    for i in range(0, h - mbSize + 1, mbSize):
        for j in range(0, w - mbSize + 1, mbSize):
            stepSize = np.int(stepMax)
            x = j
            y = i
            while stepSize >= 1:
                refBlkVerPointA = y
                refBlkHorPointA = x
                refBlkVerPointB = y
                refBlkHorPointB = x + stepSize
                refBlkVerPointC = y + stepSize
                refBlkHorPointC = x
                if _checkBounded(refBlkHorPointA, refBlkVerPointA, w, h, mbSize):
                    costs[0] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVerPointA:refBlkVerPointA + mbSize, refBlkHorPointA:refBlkHorPointA + mbSize])
                    computations += 1
                if _checkBounded(refBlkHorPointB, refBlkVerPointB, w, h, mbSize):
                    costs[1] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVerPointB:refBlkVerPointB + mbSize, refBlkHorPointB:refBlkHorPointB + mbSize])
                    computations += 1
                if _checkBounded(refBlkHorPointC, refBlkVerPointC, w, h, mbSize):
                    costs[2] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVerPointC:refBlkVerPointC + mbSize, refBlkHorPointC:refBlkHorPointC + mbSize])
                    computations += 1
                quadrant = 0
                if costs[0] >= costs[1] and costs[0] >= costs[2]:
                    quadrant = 4
                elif costs[0] >= costs[1] and costs[0] < costs[2]:
                    quadrant = 1
                elif costs[0] < costs[1] and costs[0] < costs[2]:
                    quadrant = 2
                elif costs[0] < costs[1] and costs[0] >= costs[2]:
                    quadrant = 3
                if quadrant == 1:
                    refBlkVerPointD = y - stepSize
                    refBlkHorPointD = x
                    refBlkVerPointE = y - stepSize
                    refBlkHorPointE = x + stepSize
                    if _checkBounded(refBlkHorPointD, refBlkVerPointD, w, h, mbSize):
                        costs[3] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVerPointD:refBlkVerPointD + mbSize, refBlkHorPointD:refBlkHorPointD + mbSize])
                        computations += 1
                    if _checkBounded(refBlkHorPointE, refBlkVerPointE, w, h, mbSize):
                        costs[4] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVerPointE:refBlkVerPointE + mbSize, refBlkHorPointE:refBlkHorPointE + mbSize])
                        computations += 1
                elif quadrant == 2:
                    refBlkVerPointD = y - stepSize
                    refBlkHorPointD = x
                    refBlkVerPointE = y - stepSize
                    refBlkHorPointE = x - stepSize
                    refBlkVerPointF = y
                    refBlkHorPointF = x - stepSize
                    if _checkBounded(refBlkHorPointD, refBlkVerPointD, w, h, mbSize):
                        costs[3] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVerPointD:refBlkVerPointD + mbSize, refBlkHorPointD:refBlkHorPointD + mbSize])
                        computations += 1
                    if _checkBounded(refBlkHorPointE, refBlkVerPointE, w, h, mbSize):
                        costs[4] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVerPointE:refBlkVerPointE + mbSize, refBlkHorPointE:refBlkHorPointE + mbSize])
                        computations += 1
                    if _checkBounded(refBlkHorPointF, refBlkVerPointF, w, h, mbSize):
                        costs[5] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVerPointF:refBlkVerPointF + mbSize, refBlkHorPointF:refBlkHorPointF + mbSize])
                        computations += 1
                elif quadrant == 3:
                    refBlkVerPointD = y
                    refBlkHorPointD = x - stepSize
                    refBlkVerPointE = y - stepSize
                    refBlkHorPointE = x - stepSize
                    if _checkBounded(refBlkHorPointD, refBlkVerPointD, w, h, mbSize):
                        costs[3] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVerPointD:refBlkVerPointD + mbSize, refBlkHorPointD:refBlkHorPointD + mbSize])
                        computations += 1
                    if _checkBounded(refBlkHorPointE, refBlkVerPointE, w, h, mbSize):
                        costs[4] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVerPointE:refBlkVerPointE + mbSize, refBlkHorPointE:refBlkHorPointE + mbSize])
                        computations += 1
                elif quadrant == 4:
                    refBlkVerPointD = y + stepSize
                    refBlkHorPointD = x + stepSize
                    if _checkBounded(refBlkHorPointD, refBlkVerPointD, w, h, mbSize):
                        costs[3] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[refBlkVerPointD:refBlkVerPointD + mbSize, refBlkHorPointD:refBlkHorPointD + mbSize])
                        computations += 1
                dxy = np.argmin(costs)
                cost = costs[dxy]
                if dxy == 2:
                    x = refBlkHorPointB
                    y = refBlkVerPointB
                elif dxy == 3:
                    x = refBlkHorPointC
                    y = refBlkVerPointC
                elif dxy == 4:
                    x = refBlkHorPointD
                    y = refBlkVerPointD
                elif dxy == 5:
                    x = refBlkHorPointE
                    y = refBlkVerPointE
                elif dxy == 6:
                    x = refBlkHorPointF
                    y = refBlkVerPointF
                costs[:] = 65537
                stepSize = np.int(stepSize / 2)
            vectors[np.int(i / mbSize), np.int(j / mbSize), :] = [y - i, x - j]
            costs[:] = 65537
    return (vectors, computations / (h * w / mbSize ** 2))