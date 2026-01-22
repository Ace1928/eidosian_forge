import pickle
from rdkit import Chem, DataStructs
def BuildRDKitFP(mol):
    return Chem.RDKFingerprint(mol, nBitsPerHash=1)