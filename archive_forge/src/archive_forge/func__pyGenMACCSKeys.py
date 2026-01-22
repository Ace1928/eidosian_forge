from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
def _pyGenMACCSKeys(mol, **kwargs):
    """ generates the MACCS fingerprint for a molecules

   **Arguments**

     - mol: the molecule to be fingerprinted

     - any extra keyword arguments are ignored
     
   **Returns**

      a _DataStructs.SparseBitVect_ containing the fingerprint.

  >>> m = Chem.MolFromSmiles('CNO')
  >>> bv = GenMACCSKeys(m)
  >>> tuple(bv.GetOnBits())
  (24, 68, 69, 71, 93, 94, 102, 124, 131, 139, 151, 158, 160, 161, 164)
  >>> bv = GenMACCSKeys(Chem.MolFromSmiles('CCC'))
  >>> tuple(bv.GetOnBits())
  (74, 114, 149, 155, 160)

  """
    global maccsKeys
    if maccsKeys is None:
        maccsKeys = [(None, 0)] * len(smartsPatts.keys())
        _InitKeys(maccsKeys, smartsPatts)
    ctor = kwargs.get('ctor', DataStructs.SparseBitVect)
    res = ctor(len(maccsKeys) + 1)
    for i, (patt, count) in enumerate(maccsKeys):
        if patt is not None:
            if count == 0:
                res[i + 1] = mol.HasSubstructMatch(patt)
            else:
                matches = mol.GetSubstructMatches(patt)
                if len(matches) > count:
                    res[i + 1] = 1
        elif i + 1 == 125:
            ri = mol.GetRingInfo()
            nArom = 0
            res[125] = 0
            for ring in ri.BondRings():
                isArom = True
                for bondIdx in ring:
                    if not mol.GetBondWithIdx(bondIdx).GetIsAromatic():
                        isArom = False
                        break
                if isArom:
                    nArom += 1
                    if nArom > 1:
                        res[125] = 1
                        break
        elif i + 1 == 166:
            res[166] = 0
            if len(Chem.GetMolFrags(mol)) > 1:
                res[166] = 1
    return res