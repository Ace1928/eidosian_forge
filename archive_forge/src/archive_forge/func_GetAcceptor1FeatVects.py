import math
import numpy
from rdkit import Chem, Geometry
def GetAcceptor1FeatVects(conf, featAtoms, scale=1.5):
    """
  Get the direction vectors for Acceptor of type 1

  This is a acceptor with one heavy atom neighbor. There are two possibilities we will
  consider here
  1. The bond to the heavy atom is a single bond e.g. CO
     In this case we don't know the exact direction and we just use the inversion of this bond direction
     and mark this direction as a 'cone'
  2. The bond to the heavy atom is a double bond e.g. C=O
     In this case the we have two possible direction except in some special cases e.g. SO2
     where again we will use bond direction
     
  ARGUMENTS:
    featAtoms - list of atoms that are part of the feature
    scale - length of the direction vector
  """
    assert len(featAtoms) == 1
    aid = featAtoms[0]
    mol = conf.GetOwningMol()
    nbrs = mol.GetAtomWithIdx(aid).GetNeighbors()
    cpt = conf.GetAtomPosition(aid)
    heavyAt = -1
    for nbr in nbrs:
        if nbr.GetAtomicNum() != 1:
            heavyAt = nbr
            break
    singleBnd = mol.GetBondBetweenAtoms(aid, heavyAt.GetIdx()).GetBondType() > Chem.BondType.SINGLE
    sulfur = heavyAt.GetAtomicNum() == 16
    if singleBnd or sulfur:
        v1 = conf.GetAtomPosition(heavyAt.GetIdx())
        v1 -= cpt
        v1.Normalize()
        v1 *= -1.0 * scale
        v1 += cpt
        return (((cpt, v1),), 'cone')
    hvNbrs = heavyAt.GetNeighbors()
    hvNbr = -1
    for nbr in hvNbrs:
        if nbr.GetIdx() != aid:
            hvNbr = nbr
            break
    pt1 = conf.GetAtomPosition(hvNbr.GetIdx())
    v1 = conf.GetAtomPosition(heavyAt.GetIdx())
    pt1 -= v1
    v1 -= cpt
    rotAxis = v1.CrossProduct(pt1)
    rotAxis.Normalize()
    bv1 = ArbAxisRotation(120, rotAxis, v1)
    bv1.Normalize()
    bv1 *= scale
    bv1 += cpt
    bv2 = ArbAxisRotation(-120, rotAxis, v1)
    bv2.Normalize()
    bv2 *= scale
    bv2 += cpt
    return (((cpt, bv1), (cpt, bv2)), 'linear')