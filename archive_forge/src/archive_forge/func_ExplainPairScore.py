from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.AtomPairs import Utils
from rdkit.Chem.rdMolDescriptors import (GetAtomPairFingerprint,
def ExplainPairScore(score, includeChirality=False):
    """
  >>> from rdkit import Chem
  >>> m = Chem.MolFromSmiles('C=CC')
  >>> score = pyScorePair(m.GetAtomWithIdx(0), m.GetAtomWithIdx(1), 1)
  >>> ExplainPairScore(score)
  (('C', 1, 1), 1, ('C', 2, 1))
  >>> score = pyScorePair(m.GetAtomWithIdx(0), m.GetAtomWithIdx(2), 2)
  >>> ExplainPairScore(score)
  (('C', 1, 0), 2, ('C', 1, 1))
  >>> score = pyScorePair(m.GetAtomWithIdx(1), m.GetAtomWithIdx(2), 1)
  >>> ExplainPairScore(score)
  (('C', 1, 0), 1, ('C', 2, 1))
  >>> score = pyScorePair(m.GetAtomWithIdx(2), m.GetAtomWithIdx(1), 1)
  >>> ExplainPairScore(score)
  (('C', 1, 0), 1, ('C', 2, 1))

  We can optionally deal with chirality too
  >>> m = Chem.MolFromSmiles('C[C@H](F)Cl')
  >>> score = pyScorePair(m.GetAtomWithIdx(0), m.GetAtomWithIdx(1), 1)
  >>> ExplainPairScore(score)
  (('C', 1, 0), 1, ('C', 3, 0))
  >>> score = pyScorePair(m.GetAtomWithIdx(0), m.GetAtomWithIdx(1), 1, includeChirality=True)
  >>> ExplainPairScore(score, includeChirality=True)
  (('C', 1, 0, ''), 1, ('C', 3, 0, 'R'))
  >>> m = Chem.MolFromSmiles('F[C@@H](Cl)[C@H](F)Cl')
  >>> score = pyScorePair(m.GetAtomWithIdx(1), m.GetAtomWithIdx(3), 1, includeChirality=True)
  >>> ExplainPairScore(score, includeChirality=True)
  (('C', 3, 0, 'R'), 1, ('C', 3, 0, 'S'))

  """
    codeSize = rdMolDescriptors.AtomPairsParameters.codeSize
    if includeChirality:
        codeSize += rdMolDescriptors.AtomPairsParameters.numChiralBits
    codeMask = (1 << codeSize) - 1
    pathMask = (1 << numPathBits) - 1
    dist = score & pathMask
    score = score >> numPathBits
    code1 = score & codeMask
    score = score >> codeSize
    code2 = score & codeMask
    return (Utils.ExplainAtomCode(code1, includeChirality=includeChirality), dist, Utils.ExplainAtomCode(code2, includeChirality=includeChirality))