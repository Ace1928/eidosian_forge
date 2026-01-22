import numpy
from rdkit import Chem, Geometry, RDLogger
from rdkit.Chem.Subshape import SubshapeObjects
from rdkit.Numerics import Alignment
def _checkMatchFeatures(self, targetPts, queryPts, alignment):
    nMatched = 0
    for i in range(3):
        tgtFeats = targetPts[alignment.targetTri[i]].molFeatures
        qFeats = queryPts[alignment.queryTri[i]].molFeatures
        if not tgtFeats and (not qFeats):
            nMatched += 1
        else:
            for jFeat in tgtFeats:
                if jFeat in qFeats:
                    nMatched += 1
                    break
        if nMatched >= self.numFeatThresh:
            break
    return nMatched >= self.numFeatThresh