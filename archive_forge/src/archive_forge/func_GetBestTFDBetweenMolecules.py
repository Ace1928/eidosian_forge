import math
import os
from rdkit import Chem, Geometry, RDConfig, rdBase
from rdkit.Chem import rdchem, rdMolDescriptors
def GetBestTFDBetweenMolecules(mol1, mol2, confId1=-1, useWeights=True, maxDev='equal', symmRadius=2, ignoreColinearBonds=True):
    """ Wrapper to calculate the best TFD between a single conformer of mol1 and all the conformers of mol2
      Important: The two molecules must be isomorphic

      Arguments:
      - mol1:     first instance of the molecule of interest
      - mol2:     second instance the molecule of interest
      - confId1:  conformer index for mol1 (default: first conformer)
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

      Return: TFD value
  """
    if Chem.MolToSmiles(mol1) != Chem.MolToSmiles(mol2):
        raise ValueError('The two molecules must be instances of the same molecule!')
    mol2 = _getSameAtomOrder(mol1, mol2)
    tl, tlr = CalculateTorsionLists(mol1, maxDev=maxDev, symmRadius=symmRadius, ignoreColinearBonds=ignoreColinearBonds)
    torsion1 = CalculateTorsionAngles(mol1, tl, tlr, confId=confId1)
    if useWeights:
        weights = CalculateTorsionWeights(mol1, ignoreColinearBonds=ignoreColinearBonds)
    else:
        weights = None
    best = 100000000.0
    for conf in mol2.GetConformers():
        torsion2 = CalculateTorsionAngles(mol2, tl, tlr, confId=conf.GetId())
        tfd = CalculateTFD(torsion1, torsion2, weights=weights)
        best = min(best, tfd)
    return best