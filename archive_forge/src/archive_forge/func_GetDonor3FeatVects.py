import math
import numpy
from rdkit import Chem, Geometry
def GetDonor3FeatVects(conf, featAtoms, scale=1.5):
    """
  Get the direction vectors for Donor of type 3

  This is a donor with three heavy atoms as neighbors. We will assume
  a tetrahedral arrangement of these neighbors. So the direction we are seeking
  is the last fourth arm of the sp3 arrangement

  ARGUMENTS:
    featAtoms - list of atoms that are part of the feature
    scale - length of the direction vector
  """
    assert len(featAtoms) == 1
    return (_GetTetrahedralFeatVect(conf, featAtoms[0], scale), 'linear')