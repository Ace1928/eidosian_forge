import pickle
from rdkit import Chem, DataStructs
def BuildTorsionsFP(mol):
    from rdkit.Chem.AtomPairs import Torsions
    fp = Torsions.GetTopologicalTorsionFingerprintAsIntVect(mol)
    fp._sumCache = fp.GetTotalVal()
    return fp