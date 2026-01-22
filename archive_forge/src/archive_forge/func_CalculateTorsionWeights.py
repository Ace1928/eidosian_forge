import math
import os
from rdkit import Chem, Geometry, RDConfig, rdBase
from rdkit.Chem import rdchem, rdMolDescriptors
def CalculateTorsionWeights(mol, aid1=-1, aid2=-1, ignoreColinearBonds=True):
    """ Calculate the weights for the torsions in a molecule.
      By default, the highest weight is given to the bond 
      connecting the two most central atoms.
      If desired, two alternate atoms can be specified (must 
      be connected by a bond).

      Arguments:
      - mol:   the molecule of interest
      - aid1:  index of the first atom (default: most central)
      - aid2:  index of the second atom (default: second most central)
      - ignoreColinearBonds: if True (default), single bonds adjacent to
                             triple bonds are ignored
                             if False, alternative not-covalently bound
                             atoms are used to define the torsion

      Return: list of torsion weights (both non-ring and ring)
  """
    distmat = Chem.GetDistanceMatrix(mol)
    if aid1 < 0 and aid2 < 0:
        aid1, aid2 = _findCentralBond(mol, distmat)
    else:
        b = mol.GetBondBetweenAtoms(aid1, aid2)
        if b is None:
            raise ValueError('Specified atoms must be connected by a bond.')
    beta = _calculateBeta(mol, distmat, aid1)
    bonds = _getBondsForTorsions(mol, ignoreColinearBonds)
    weights = []
    for bid1, bid2, nb1, nb2 in bonds:
        if (bid1, bid2) == (aid1, aid2) or (bid2, bid1) == (aid1, aid2):
            d = 0
        else:
            d = min(distmat[aid1][bid1], distmat[aid1][bid2], distmat[aid2][bid1], distmat[aid2][bid2]) + 1
        w = math.exp(-beta * (d * d))
        weights.append(w)
    rings = mol.GetRingInfo()
    for r in rings.BondRings():
        tmp = []
        num = len(r)
        for bidx in r:
            b = mol.GetBondWithIdx(bidx)
            bid1 = b.GetBeginAtomIdx()
            bid2 = b.GetEndAtomIdx()
            d = min(distmat[aid1][bid1], distmat[aid1][bid2], distmat[aid2][bid1], distmat[aid2][bid2]) + 1
            tmp.append(d)
        w = sum(tmp) / float(num)
        w = math.exp(-beta * (w * w))
        weights.append(w * (num / 2.0))
    return weights