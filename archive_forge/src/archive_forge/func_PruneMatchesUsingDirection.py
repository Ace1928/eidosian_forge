import numpy
from rdkit import Chem, Geometry, RDLogger
from rdkit.Chem.Subshape import SubshapeObjects
from rdkit.Numerics import Alignment
def PruneMatchesUsingDirection(self, target, query, alignments, pruneStats=None):
    i = 0
    tgtPts = target.skelPts
    queryPts = query.skelPts
    while i < len(alignments):
        if not self._checkMatchDirections(tgtPts, queryPts, alignments[i]):
            if pruneStats is not None:
                pruneStats['direction'] = pruneStats.get('direction', 0) + 1
            del alignments[i]
        else:
            i += 1