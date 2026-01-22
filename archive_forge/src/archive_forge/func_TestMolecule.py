import sys
from rdkit import Chem
from rdkit.Chem import Randomize
def TestMolecule(mol):
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
    except ValueError as msg:
        return -1
    except Exception:
        import traceback
        traceback.print_exc()
        return -2
    if mol.GetNumAtoms():
        try:
            Randomize.CheckCanonicalization(mol, 10)
        except Exception:
            import traceback
            traceback.print_exc()
            return -3
    return 0