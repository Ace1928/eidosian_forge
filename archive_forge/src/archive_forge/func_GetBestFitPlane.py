import numpy as np
from numpy import linalg
from rdkit import Chem
from rdkit.Chem import AllChem
def GetBestFitPlane(pts, weights=None):
    if weights is None:
        wSum = len(pts)
        origin = np.sum(pts, 0)
    origin /= wSum
    sumXX = 0
    sumXY = 0
    sumXZ = 0
    sumYY = 0
    sumYZ = 0
    sumZZ = 0
    sums = np.zeros((3, 3), np.double)
    for pt in pts:
        dp = pt - origin
        for i in range(3):
            sums[i, i] += dp[i] * dp[i]
            for j in range(i + 1, 3):
                sums[i, j] += dp[i] * dp[j]
                sums[j, i] += dp[i] * dp[j]
    sums /= wSum
    vals, vects = linalg.eigh(sums)
    order = np.argsort(vals)
    normal = vects[:, order[0]]
    plane = np.zeros((4,), np.double)
    plane[:3] = normal
    plane[3] = -1 * normal.dot(origin)
    return plane