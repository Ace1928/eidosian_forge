from rdkit import RDLogger as logging
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen
from rdkit.Chem.ChemUtils.AlignDepict import AlignDepict
def MoveDummyNeighborsToBeginning(mol, useAll=False):
    dummyPatt = Chem.MolFromSmiles('[*]')
    matches = mol.GetSubstructMatches(dummyPatt)
    res = []
    for match in matches:
        smi = Chem.MolToSmiles(mol, rootedAtAtom=match[0])
        entry = Chem.MolFromSmiles(smi)
        entry = Chem.DeleteSubstructs(entry, dummyPatt)
        for propN in mol.GetPropNames():
            entry.SetProp(propN, mol.GetProp(propN))
        res.append(entry)
        if not useAll:
            break
    return res