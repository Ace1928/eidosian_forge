import math
import os
from rdkit import Chem, Geometry, RDConfig, rdBase
from rdkit.Chem import rdchem, rdMolDescriptors
def _findCentralBond(mol, distmat):
    """ Helper function to identify the atoms of the most central bond.

      Arguments:
      - mol:     the molecule of interest
      - distmat: distance matrix of the molecule

      Return: atom indices of the two most central atoms (in order)
  """
    from numpy import std
    stds = []
    for i in range(mol.GetNumAtoms()):
        if len(_getHeavyAtomNeighbors(mol.GetAtomWithIdx(i))) < 2:
            continue
        tmp = [d for d in distmat[i]]
        tmp.pop(i)
        stds.append((std(tmp), i))
    stds.sort()
    aid1 = stds[0][1]
    i = 1
    while 1:
        if mol.GetBondBetweenAtoms(aid1, stds[i][1]) is None:
            i += 1
        else:
            aid2 = stds[i][1]
            break
    return (aid1, aid2)