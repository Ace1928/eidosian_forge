from rdkit import Chem
from rdkit.Chem import AllChem
def _pyGetScaffoldForMol(mol):
    while mol.HasSubstructMatch(murckoQ):
        for patt in murckoPatts:
            mol = Chem.DeleteSubstructs(mol, patt)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            if atom.GetNoImplicit() and atom.GetExplicitValence() < 4:
                atom.SetNoImplicit(False)
    h = Chem.MolFromSmiles('[H]')
    mol = Chem.ReplaceSubstructs(mol, Chem.MolFromSmarts('[D1;$([D1]-n)]'), h, True)[0]
    mol = Chem.RemoveHs(mol)
    return mol