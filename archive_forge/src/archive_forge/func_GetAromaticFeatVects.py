import math
import numpy
from rdkit import Chem, Geometry
def GetAromaticFeatVects(conf, featAtoms, featLoc, scale=1.5):
    """
  Compute the direction vector for an aromatic feature
  
  ARGUMENTS:
     conf - a conformer
     featAtoms - list of atom IDs that make up the feature
     featLoc - location of the aromatic feature specified as point3d
     scale - the size of the direction vector
  """
    dirType = 'linear'
    head = featLoc
    ats = [conf.GetAtomPosition(x) for x in featAtoms]
    v1 = ats[0] - head
    v2 = ats[1] - head
    norm1 = v1.CrossProduct(v2)
    norm1.Normalize()
    norm1 *= scale
    norm2 = head - norm1
    norm1 += head
    return (((head, norm1), (head, norm2)), dirType)