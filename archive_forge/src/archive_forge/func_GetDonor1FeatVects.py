import math
import numpy
from rdkit import Chem, Geometry
def GetDonor1FeatVects(conf, featAtoms, scale=1.5):
    """
  Get the direction vectors for Donor of type 1

  This is a donor with one heavy atom. It is not clear where we should we should be putting the
  direction vector for this. It should probably be a cone. In this case we will just use the
  direction vector from the donor atom to the heavy atom
    
  ARGUMENTS:
    
    featAtoms - list of atoms that are part of the feature
    scale - length of the direction vector
  """
    assert len(featAtoms) == 1
    aid = featAtoms[0]
    mol = conf.GetOwningMol()
    nbrs = mol.GetAtomWithIdx(aid).GetNeighbors()
    hnbr = -1
    for nbr in nbrs:
        if nbr.GetAtomicNum() != 1:
            hnbr = nbr.GetIdx()
            break
    cpt = conf.GetAtomPosition(aid)
    v1 = conf.GetAtomPosition(hnbr)
    v1 -= cpt
    v1.Normalize()
    v1 *= -1.0 * scale
    v1 += cpt
    return (((cpt, v1),), 'cone')