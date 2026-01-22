import bisect
import numpy
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors, rdPartialCharges
def _pyLabuteHelper(mol, includeHs=1, force=0):
    """ *Internal Use Only*
    helper function for LabuteASA calculation
    returns an array of atomic contributions to the ASA

  **Note:** Changes here affect the version numbers of all ASA descriptors

  """
    import math
    if not force:
        try:
            res = mol._labuteContribs
        except AttributeError:
            pass
        else:
            if res.all():
                return res
    nAts = mol.GetNumAtoms()
    Vi = numpy.zeros(nAts + 1, 'd')
    rads = numpy.zeros(nAts + 1, 'd')
    rads[0] = ptable.GetRb0(1)
    for i in range(nAts):
        rads[i + 1] = ptable.GetRb0(mol.GetAtomWithIdx(i).GetAtomicNum())
    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx() + 1
        idx2 = bond.GetEndAtomIdx() + 1
        Ri = rads[idx1]
        Rj = rads[idx2]
        if not bond.GetIsAromatic():
            bij = Ri + Rj - bondScaleFacts[bond.GetBondType()]
        else:
            bij = Ri + Rj - bondScaleFacts[0]
        dij = min(max(abs(Ri - Rj), bij), Ri + Rj)
        Vi[idx1] += Rj * Rj - (Ri - dij) ** 2 / dij
        Vi[idx2] += Ri * Ri - (Rj - dij) ** 2 / dij
    if includeHs:
        j = 0
        Rj = rads[j]
        for i in range(1, nAts + 1):
            Ri = rads[i]
            bij = Ri + Rj
            dij = min(max(abs(Ri - Rj), bij), Ri + Rj)
            Vi[i] += Rj * Rj - (Ri - dij) ** 2 / dij
            Vi[j] += Ri * Ri - (Rj - dij) ** 2 / dij
    for i in range(nAts + 1):
        Ri = rads[i]
        Vi[i] = 4 * math.pi * Ri ** 2 - math.pi * Ri * Vi[i]
    mol._labuteContribs = Vi
    return Vi