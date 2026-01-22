import math
import numpy
from rdkit import Chem, Geometry
def GetAcceptor2FeatVects(conf, featAtoms, scale=1.5):
    """
  Get the direction vectors for Acceptor of type 2
  
  This is the acceptor with two adjacent heavy atoms. We will special case a few things here.
  If the acceptor atom is an oxygen we will assume a sp3 hybridization
  the acceptor directions (two of them)
  reflect that configurations. Otherwise the direction vector in plane with the neighboring
  heavy atoms
  
  ARGUMENTS:
      featAtoms - list of atoms that are part of the feature
      scale - length of the direction vector
  """
    assert len(featAtoms) == 1
    aid = featAtoms[0]
    cpt = conf.GetAtomPosition(aid)
    mol = conf.GetOwningMol()
    nbrs = list(mol.GetAtomWithIdx(aid).GetNeighbors())
    hydrogens = []
    tmp = []
    while len(nbrs):
        nbr = nbrs.pop()
        if nbr.GetAtomicNum() == 1:
            hydrogens.append(nbr)
        else:
            tmp.append(nbr)
    nbrs = tmp
    assert len(nbrs) == 2
    bvec = _findAvgVec(conf, cpt, nbrs)
    bvec *= -1.0 * scale
    if mol.GetAtomWithIdx(aid).GetAtomicNum() == 8:
        v1 = conf.GetAtomPosition(nbrs[0].GetIdx())
        v1 -= cpt
        v2 = conf.GetAtomPosition(nbrs[1].GetIdx())
        v2 -= cpt
        rotAxis = v1 - v2
        rotAxis.Normalize()
        bv1 = ArbAxisRotation(54.5, rotAxis, bvec)
        bv1 += cpt
        bv2 = ArbAxisRotation(-54.5, rotAxis, bvec)
        bv2 += cpt
        return (((cpt, bv1), (cpt, bv2)), 'linear')
    bvec += cpt
    return (((cpt, bvec),), 'linear')