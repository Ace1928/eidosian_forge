import copy
import numpy
from rdkit.Chem.Pharm2D import Utils
from rdkit.DataStructs import (IntSparseIntVect, LongSparseIntVect,
def GetBitIdx(self, featIndices, dists, sortIndices=True):
    """ returns the index for a pharmacophore described using a set of
          feature indices and distances

        **Arguments***

          - featIndices: a sequence of feature indices

          - dists: a sequence of distance between the features, only the
            unique distances should be included, and they should be in the
            order defined in Utils.

          - sortIndices : sort the indices

        **Returns**

          the integer bit index

        """
    nPoints = len(featIndices)
    if nPoints > 3:
        raise NotImplementedError('>3 points not supported')
    if nPoints < self.minPointCount:
        raise IndexError('bad number of points')
    if nPoints > self.maxPointCount:
        raise IndexError('bad number of points')
    startIdx = self._starts[nPoints]
    if sortIndices:
        tmp = list(featIndices)
        tmp.sort()
        featIndices = tmp
    if featIndices[0] < 0:
        raise IndexError('bad feature index')
    if max(featIndices) >= self._nFeats:
        raise IndexError('bad feature index')
    if nPoints == 3:
        featIndices, dists = Utils.OrderTriangle(featIndices, dists)
    offset = Utils.CountUpTo(self._nFeats, nPoints, featIndices)
    if _verbose:
        print(f'offset for feature {str(featIndices)}: {offset}')
    offset *= len(self._scaffolds[len(dists)])
    try:
        if _verbose:
            print('>>>>>>>>>>>>>>>>>>>>>>>')
            print('\tScaffolds:', repr(self._scaffolds[len(dists)]), type(self._scaffolds[len(dists)]))
            print('\tDists:', repr(dists), type(dists))
            print('\tbins:', repr(self._bins), type(self._bins))
        bin_ = self._findBinIdx(dists, self._bins, self._scaffolds[len(dists)])
    except ValueError:
        fams = self.GetFeatFamilies()
        fams = [fams[x] for x in featIndices]
        raise IndexError('distance bin not found: feats: %s; dists=%s; bins=%s; scaffolds: %s' % (fams, dists, self._bins, self._scaffolds))
    return startIdx + offset + bin_