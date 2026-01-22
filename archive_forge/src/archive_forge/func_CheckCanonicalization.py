from rdkit import Chem
from rdkit import RDRandom as random
def CheckCanonicalization(mol, nReps=10):
    refSmi = Chem.MolToSmiles(mol, False)
    for i in range(nReps):
        m2 = RandomizeMol(mol)
        smi = Chem.MolToSmiles(m2, False)
        if smi != refSmi:
            raise ValueError('\nRef: %s\n   : %s' % (refSmi, smi))