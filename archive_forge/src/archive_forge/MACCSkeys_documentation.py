from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
 generates the MACCS fingerprint for a molecules

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

  