import pickle
from rdkit import Chem, DataStructs
def BuildAtomPairFP(mol):
    from rdkit.Chem.AtomPairs import Pairs
    fp = Pairs.GetAtomPairFingerprintAsIntVect(mol)
    fp._sumCache = fp.GetTotalVal()
    return fp