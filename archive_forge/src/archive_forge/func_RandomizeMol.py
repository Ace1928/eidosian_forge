from rdkit import Chem
from rdkit import RDRandom as random
def RandomizeMol(mol):
    mb = Chem.MolToMolBlock(mol)
    mb = RandomizeMolBlock(mb)
    return Chem.MolFromMolBlock(mb)