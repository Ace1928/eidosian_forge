import copy
import math
import numpy
from rdkit import Chem, DataStructs, Geometry
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.Draw import rdMolDraw2D
def GetRDKFingerprint(mol, atomId=-1, fpType='bv', nBits=2048, minPath=1, maxPath=5, nBitsPerHash=2, **kwargs):
    """
    Calculates the RDKit fingerprint with the paths of atomId removed.

    Parameters:
      mol -- the molecule of interest
      atomId -- the atom to remove the paths for (if -1, no path is removed)
      fpType -- the type of RDKit fingerprint: 'bv'
      nBits -- the size of the bit vector
      minPath -- minimum path length
      maxPath -- maximum path length
      nBitsPerHash -- number of to set per path
    """
    if fpType not in ['bv', '']:
        raise ValueError('Unknown RDKit fingerprint type')
    fpType = 'bv'
    if not hasattr(mol, '_fpInfo'):
        info = []
        molFp = Chem.RDKFingerprint(mol, fpSize=nBits, minPath=minPath, maxPath=maxPath, nBitsPerHash=nBitsPerHash, atomBits=info, **kwargs)
        mol._fpInfo = (molFp, info)
    if atomId < 0:
        return mol._fpInfo[0]
    if atomId >= mol.GetNumAtoms():
        raise ValueError('atom index greater than number of atoms')
    if len(mol._fpInfo) != 2:
        raise ValueError('_fpInfo not set')
    molFp = copy.deepcopy(mol._fpInfo[0])
    molFp.UnSetBitsFromList(mol._fpInfo[1][atomId])
    return molFp