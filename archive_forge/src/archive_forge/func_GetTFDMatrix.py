import math
import os
from rdkit import Chem, Geometry, RDConfig, rdBase
from rdkit.Chem import rdchem, rdMolDescriptors
def GetTFDMatrix(mol, useWeights=True, maxDev='equal', symmRadius=2, ignoreColinearBonds=True):
    """ Wrapper to calculate the matrix of TFD values for the
      conformers of a molecule.

      Arguments:
      - mol:      the molecule of interest
      - useWeights: flag for using torsion weights in the TFD calculation
      - maxDev:   maximal deviation used for normalization
                  'equal': all torsions are normalized using 180.0 (default)
                  'spec':  each torsion is normalized using its specific
                           maximal deviation as given in the paper
      - symmRadius: radius used for calculating the atom invariants
                    (default: 2)
      - ignoreColinearBonds: if True (default), single bonds adjacent to
                             triple bonds are ignored
                             if False, alternative not-covalently bound
                             atoms are used to define the torsion

      Return: matrix of TFD values
      Note that the returned matrix is symmetrical, i.e. it is the
      lower half of the matrix, e.g. for 5 conformers:
      matrix = [ a,
                 b, c,
                 d, e, f,
                 g, h, i, j]
  """
    tl, tlr = CalculateTorsionLists(mol, maxDev=maxDev, symmRadius=symmRadius, ignoreColinearBonds=ignoreColinearBonds)
    numconf = mol.GetNumConformers()
    torsions = [CalculateTorsionAngles(mol, tl, tlr, confId=conf.GetId()) for conf in mol.GetConformers()]
    tfdmat = []
    if useWeights:
        weights = CalculateTorsionWeights(mol, ignoreColinearBonds=ignoreColinearBonds)
    else:
        weights = None
    for i in range(0, numconf):
        for j in range(0, i):
            tfdmat.append(CalculateTFD(torsions[i], torsions[j], weights=weights))
    return tfdmat