import copy
import math
import numpy
from rdkit import Chem, DataStructs, Geometry
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.Draw import rdMolDraw2D
def GetMorganFingerprint(mol, atomId=-1, radius=2, fpType='bv', nBits=2048, useFeatures=False, **kwargs):
    """
    Calculates the Morgan fingerprint with the environments of atomId removed.

    Parameters:
      mol -- the molecule of interest
      radius -- the maximum radius
      fpType -- the type of Morgan fingerprint: 'count' or 'bv'
      atomId -- the atom to remove the environments for (if -1, no environments is removed)
      nBits -- the size of the bit vector (only for fpType = 'bv')
      useFeatures -- if false: ConnectivityMorgan, if true: FeatureMorgan

    any additional keyword arguments will be passed to the fingerprinting function.
    """
    if fpType not in ['bv', 'count']:
        raise ValueError('Unknown Morgan fingerprint type')
    isBitVect = fpType == 'bv'
    if not hasattr(mol, '_fpInfo'):
        info = {}
        if isBitVect:
            molFp = rdMD.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=useFeatures, bitInfo=info, **kwargs)
            bitmap = [DataStructs.ExplicitBitVect(nBits) for _ in range(mol.GetNumAtoms())]
        else:
            molFp = rdMD.GetMorganFingerprint(mol, radius, useFeatures=useFeatures, bitInfo=info, **kwargs)
            bitmap = [[] for _ in range(mol.GetNumAtoms())]
        for bit, es in info.items():
            for at1, rad in es:
                if rad == 0:
                    if isBitVect:
                        bitmap[at1][bit] = 1
                    else:
                        bitmap[at1].append(bit)
                else:
                    env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, at1)
                    amap = {}
                    Chem.PathToSubmol(mol, env, atomMap=amap)
                    for at2 in amap.keys():
                        if isBitVect:
                            bitmap[at2][bit] = 1
                        else:
                            bitmap[at2].append(bit)
        mol._fpInfo = (molFp, bitmap)
    if atomId < 0:
        return mol._fpInfo[0]
    if atomId >= mol.GetNumAtoms():
        raise ValueError('atom index greater than number of atoms')
    if len(mol._fpInfo) != 2:
        raise ValueError('_fpInfo not set')
    if isBitVect:
        molFp = mol._fpInfo[0] ^ mol._fpInfo[1][atomId]
    else:
        molFp = copy.deepcopy(mol._fpInfo[0])
        for bit in mol._fpInfo[1][atomId]:
            molFp[bit] -= 1
    return molFp