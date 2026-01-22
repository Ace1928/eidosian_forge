import numpy
from rdkit import Chem, Geometry, RDLogger
from rdkit.Chem.Subshape import SubshapeObjects
from rdkit.Numerics import Alignment
def _checkMatchDirections(self, targetPts, queryPts, alignment):
    dot = 0.0
    for i in range(3):
        tgtPt = targetPts[alignment.targetTri[i]]
        queryPt = queryPts[alignment.queryTri[i]]
        qv = queryPt.shapeDirs[0]
        tv = tgtPt.shapeDirs[0]
        rotV = [0.0] * 3
        rotV[0] = alignment.transform[0, 0] * qv[0] + alignment.transform[0, 1] * qv[1] + alignment.transform[0, 2] * qv[2]
        rotV[1] = alignment.transform[1, 0] * qv[0] + alignment.transform[1, 1] * qv[1] + alignment.transform[1, 2] * qv[2]
        rotV[2] = alignment.transform[2, 0] * qv[0] + alignment.transform[2, 1] * qv[1] + alignment.transform[2, 2] * qv[2]
        dot += abs(rotV[0] * tv[0] + rotV[1] * tv[1] + rotV[2] * tv[2])
        if dot >= self.dirThresh:
            break
    alignment.dirMatch = dot
    return dot >= self.dirThresh