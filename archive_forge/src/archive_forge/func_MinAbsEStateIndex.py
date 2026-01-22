import numpy
from rdkit import Chem
def MinAbsEStateIndex(mol, force=1):
    return min((abs(x) for x in EStateIndices(mol, force)))