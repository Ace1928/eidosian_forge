import math
import sys
import time
import numpy
import rdkit.DistanceGeometry as DG
from rdkit import Chem
from rdkit import RDLogger as logging
from rdkit.Chem import ChemicalFeatures, ChemicalForceFields
from rdkit.Chem import rdDistGeom as MolDG
from rdkit.Chem.Pharm3D import ExcludedVolume
from rdkit.ML.Data import Stats
def AddExcludedVolumes(bm, excludedVolumes, smoothIt=True):
    """ Adds a set of excluded volumes to the bounds matrix
  and returns the new matrix

  excludedVolumes is a list of ExcludedVolume objects


   >>> boundsMat = numpy.array([[0.0, 2.0, 2.0],[1.0, 0.0, 2.0],[1.0, 1.0, 0.0]])
   >>> ev1 = ExcludedVolume.ExcludedVolume(([(0, ), 0.5, 1.0], ), exclusionDist=1.5)
   >>> bm = AddExcludedVolumes(boundsMat, (ev1, ))

   the results matrix is one bigger:

   >>> bm.shape == (4, 4)
   True

   and the original bounds mat is not altered:

   >>> boundsMat.shape == (3, 3)
   True

   >>> print(', '.join([f'{x:.3f}' for x in bm[-1]]))
   0.500, 1.500, 1.500, 0.000
   >>> print(', '.join([f'{x:.3f}' for x in bm[:,-1]]))
   1.000, 3.000, 3.000, 0.000

  """
    oDim = bm.shape[0]
    dim = oDim + len(excludedVolumes)
    res = numpy.zeros((dim, dim), dtype=numpy.float64)
    res[:oDim, :oDim] = bm
    for i, vol in enumerate(excludedVolumes):
        bmIdx = oDim + i
        vol.index = bmIdx
        res[bmIdx, :bmIdx] = vol.exclusionDist
        res[:bmIdx, bmIdx] = 1000.0
        for indices, minV, maxV in vol.featInfo:
            for index in indices:
                try:
                    res[bmIdx, index] = minV
                    res[index, bmIdx] = maxV
                except IndexError:
                    logger.error(f'BAD INDEX: res[{bmIdx},{index}], shape is {str(res.shape)}')
                    raise IndexError
        for j in range(bmIdx + 1, dim):
            res[bmIdx, j:dim] = 0.0
            res[j:dim, bmIdx] = 1000.0
    if smoothIt:
        DG.DoTriangleSmoothing(res)
    return res