from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.AtomPairs import Utils
from rdkit.Chem.rdMolDescriptors import (GetAtomPairFingerprint,
 Returns the Atom-pair fingerprint for a molecule as
  a SparseBitVect. Note that this doesn't match the standard
  definition of atom pairs, which uses counts of the
  pairs, not just their presence.

  **Arguments**:

    - mol: a molecule

  **Returns**: a SparseBitVect

  >>> from rdkit import Chem
  >>> m = Chem.MolFromSmiles('CCC')
  >>> v = [pyScorePair(m.GetAtomWithIdx(0), m.GetAtomWithIdx(1), 1),
  ...      pyScorePair(m.GetAtomWithIdx(0), m.GetAtomWithIdx(2), 2),
  ...     ]
  >>> v.sort()
  >>> fp = GetAtomPairFingerprintAsBitVect(m)
  >>> list(fp.GetOnBits()) == v
  True

  