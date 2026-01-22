import itertools
from rdkit import Chem
def SATISTypes(mol, neighborsToInclude=4):
    """ returns SATIS codes for all atoms in a molecule

   The SATIS definition used is from:
   J. Chem. Inf. Comput. Sci. _39_ 751-757 (1999)

   each SATIS code is a string consisting of _neighborsToInclude_ + 1
   2 digit numbers

   **Arguments**

     - mol: a molecule

     - neighborsToInclude (optional): the number of neighbors to include
       in the SATIS codes

   **Returns**

     a list of strings nAtoms long

  """
    nAtoms = mol.GetNumAtoms()
    atoms = mol.GetAtoms()
    atomicNums = [atom.GetAtomicNum() for atom in atoms]
    specialCaseMatches = []
    for patt, specialCaseIdx in specialCases:
        matches = mol.GetSubstructMatches(patt)
        if matches:
            matches = set(itertools.chain(*matches))
            specialCaseMatches.append((specialCaseIdx, matches))
    codes = [None] * nAtoms
    for i, atom in enumerate(atoms):
        code = [99] * (neighborsToInclude + 1)
        code[0] = min(atom.GetAtomicNum(), 99)
        otherIndices = [x.GetIdx() for x in atom.GetNeighbors()]
        otherNums = sorted([atomicNums[x] for x in otherIndices] + [1] * atom.GetTotalNumHs())
        if len(otherNums) > neighborsToInclude:
            otherNums = otherNums[-neighborsToInclude:]
        for j, otherNum in enumerate(otherNums, 1):
            code[j] = min(otherNum, 99)
        if len(otherNums) < neighborsToInclude and code[0] in [6, 8]:
            atomIdx = atom.GetIdx()
            for specialCaseIdx, matches in specialCaseMatches:
                if atomIdx in matches:
                    code[-1] = specialCaseIdx
                    break
        codes[i] = ''.join(('%02d' % x for x in code))
    return codes