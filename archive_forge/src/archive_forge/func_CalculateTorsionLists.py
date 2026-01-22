import math
import os
from rdkit import Chem, Geometry, RDConfig, rdBase
from rdkit.Chem import rdchem, rdMolDescriptors
def CalculateTorsionLists(mol, maxDev='equal', symmRadius=2, ignoreColinearBonds=True):
    """ Calculate a list of torsions for a given molecule. For each torsion
      the four atom indices are determined and stored in a set.

      Arguments:
      - mol:      the molecule of interest
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

      Return: two lists of torsions: non-ring and ring torsions
  """
    if maxDev not in ['equal', 'spec']:
        raise ValueError('maxDev must be either equal or spec')
    bonds = _getBondsForTorsions(mol, ignoreColinearBonds)
    if symmRadius > 0:
        inv = _getAtomInvariantsWithRadius(mol, symmRadius)
    else:
        inv = rdMolDescriptors.GetConnectivityInvariants(mol)
    tors_list = []
    for a1, a2, nb1, nb2 in bonds:
        d1 = _getIndexforTorsion(nb1, inv)
        d2 = _getIndexforTorsion(nb2, inv)
        if len(d1) == 1 and len(d2) == 1:
            tors_list.append(([(d1[0].GetIdx(), a1, a2, d2[0].GetIdx())], 180.0))
        elif len(d1) == 1:
            if len(nb2) == 2:
                tors_list.append(([(d1[0].GetIdx(), a1, a2, nb.GetIdx()) for nb in d2], 90.0))
            else:
                tors_list.append(([(d1[0].GetIdx(), a1, a2, nb.GetIdx()) for nb in d2], 60.0))
        elif len(d2) == 1:
            if len(nb1) == 2:
                tors_list.append(([(nb.GetIdx(), a1, a2, d2[0].GetIdx()) for nb in d1], 90.0))
            else:
                tors_list.append(([(nb.GetIdx(), a1, a2, d2[0].GetIdx()) for nb in d1], 60.0))
        else:
            tmp = []
            for n1 in d1:
                for n2 in d2:
                    tmp.append((n1.GetIdx(), a1, a2, n2.GetIdx()))
            if len(nb1) == 2 and len(nb2) == 2:
                tors_list.append((tmp, 90.0))
            elif len(nb1) == 3 and len(nb2) == 3:
                tors_list.append((tmp, 60.0))
            else:
                tors_list.append((tmp, 30.0))
    if maxDev == 'equal':
        tors_list = [(t, 180.0) for t, d in tors_list]
    rings = Chem.GetSymmSSSR(mol)
    tors_list_rings = []
    for r in rings:
        tmp = []
        num = len(r)
        if 14 <= num:
            maxdev = 180.0
        else:
            maxdev = 180.0 * math.exp(-0.025 * (num - 14) * (num - 14))
        for i in range(len(r)):
            tmp.append((r[i], r[(i + 1) % num], r[(i + 2) % num], r[(i + 3) % num]))
        tors_list_rings.append((tmp, maxdev))
    return (tors_list, tors_list_rings)