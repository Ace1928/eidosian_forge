import math
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
def ExplainAtomCode(code, branchSubtract=0, includeChirality=False):
    """

    **Arguments**:

      - the code to be considered

      - branchSubtract: (optional) the constant that was subtracted off
        the number of neighbors before integrating it into the code.
        This is used by the topological torsions code.

      - includeChirality: (optional) Determines whether or not chirality
        was included when generating the atom code.

    >>> m = Chem.MolFromSmiles('C=CC(=O)O')
    >>> code = GetAtomCode(m.GetAtomWithIdx(0))
    >>> ExplainAtomCode(code)
    ('C', 1, 1)
    >>> code = GetAtomCode(m.GetAtomWithIdx(1))
    >>> ExplainAtomCode(code)
    ('C', 2, 1)
    >>> code = GetAtomCode(m.GetAtomWithIdx(2))
    >>> ExplainAtomCode(code)
    ('C', 3, 1)
    >>> code = GetAtomCode(m.GetAtomWithIdx(3))
    >>> ExplainAtomCode(code)
    ('O', 1, 1)
    >>> code = GetAtomCode(m.GetAtomWithIdx(4))
    >>> ExplainAtomCode(code)
    ('O', 1, 0)

    we can do chirality too, that returns an extra element in the tuple:

    >>> m = Chem.MolFromSmiles('C[C@H](F)Cl')
    >>> code = GetAtomCode(m.GetAtomWithIdx(1))
    >>> ExplainAtomCode(code)
    ('C', 3, 0)
    >>> code = GetAtomCode(m.GetAtomWithIdx(1),includeChirality=True)
    >>> ExplainAtomCode(code,includeChirality=True)
    ('C', 3, 0, 'R')

    note that if we don't ask for chirality, we get the right answer even if
    the atom code was calculated with chirality:

    >>> ExplainAtomCode(code)
    ('C', 3, 0)

    non-chiral atoms return '' in the 4th field:

    >>> code = GetAtomCode(m.GetAtomWithIdx(0),includeChirality=True)
    >>> ExplainAtomCode(code,includeChirality=True)
    ('C', 1, 0, '')

    Obviously switching the chirality changes the results:

    >>> m = Chem.MolFromSmiles('C[C@@H](F)Cl')
    >>> code = GetAtomCode(m.GetAtomWithIdx(1),includeChirality=True)
    >>> ExplainAtomCode(code,includeChirality=True)
    ('C', 3, 0, 'S')

    """
    typeMask = (1 << rdMolDescriptors.AtomPairsParameters.numTypeBits) - 1
    branchMask = (1 << rdMolDescriptors.AtomPairsParameters.numBranchBits) - 1
    piMask = (1 << rdMolDescriptors.AtomPairsParameters.numPiBits) - 1
    chiMask = (1 << rdMolDescriptors.AtomPairsParameters.numChiralBits) - 1
    nBranch = int(code & branchMask)
    code = code >> rdMolDescriptors.AtomPairsParameters.numBranchBits
    nPi = int(code & piMask)
    code = code >> rdMolDescriptors.AtomPairsParameters.numPiBits
    typeIdx = int(code & typeMask)
    if typeIdx < len(rdMolDescriptors.AtomPairsParameters.atomTypes):
        atomNum = rdMolDescriptors.AtomPairsParameters.atomTypes[typeIdx]
        atomSymbol = Chem.GetPeriodicTable().GetElementSymbol(atomNum)
    else:
        atomSymbol = 'X'
    if not includeChirality:
        return (atomSymbol, nBranch, nPi)
    code = code >> rdMolDescriptors.AtomPairsParameters.numTypeBits
    chiDict = {0: '', 1: 'R', 2: 'S'}
    chiCode = int(code & chiMask)
    return (atomSymbol, nBranch, nPi, chiDict[chiCode])