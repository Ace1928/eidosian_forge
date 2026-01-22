import copy
import math
import numpy
from rdkit import Chem, DataStructs, Geometry
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.Draw import rdMolDraw2D
def GetTTFingerprint(mol, atomId=-1, fpType='normal', nBits=2048, targetSize=4, nBitsPerEntry=4, **kwargs):
    """
    Calculates the topological torsion fingerprint with the pairs of atomId removed.

    Parameters:
      mol -- the molecule of interest
      atomId -- the atom to remove the torsions for (if -1, no torsion is removed)
      fpType -- the type of TT fingerprint ('normal', 'hashed', 'bv')
      nBits -- the size of the bit vector (only for fpType='bv')
      minLength -- the minimum path length for an atom pair
      maxLength -- the maxmimum path length for an atom pair
      nBitsPerEntry -- the number of bits available for each torsion

    any additional keyword arguments will be passed to the fingerprinting function.

    """
    if fpType not in ['normal', 'hashed', 'bv']:
        raise ValueError('Unknown Topological torsion fingerprint type')
    if atomId < 0:
        return ttDict[fpType](mol, nBits, targetSize, nBitsPerEntry, 0, **kwargs)
    if atomId >= mol.GetNumAtoms():
        raise ValueError('atom index greater than number of atoms')
    return ttDict[fpType](mol, nBits, targetSize, nBitsPerEntry, [atomId], **kwargs)