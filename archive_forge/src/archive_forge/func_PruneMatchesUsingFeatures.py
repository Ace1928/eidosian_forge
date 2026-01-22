import numpy
from rdkit import Chem, Geometry, RDLogger
from rdkit.Chem.Subshape import SubshapeObjects
from rdkit.Numerics import Alignment
def PruneMatchesUsingFeatures(self, target, query, alignments, pruneStats=None):
    i = 0
    targetPts = target.skelPts
    queryPts = query.skelPts
    while i < len(alignments):
        alg = alignments[i]
        if not self._checkMatchFeatures(targetPts, queryPts, alg):
            if pruneStats is not None:
                pruneStats['features'] = pruneStats.get('features', 0) + 1
            del alignments[i]
        else:
            i += 1