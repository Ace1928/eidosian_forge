import math
import os
from rdkit import Chem, Geometry, RDConfig, rdBase
from rdkit.Chem import rdchem, rdMolDescriptors
def _getSameAtomOrder(mol1, mol2):
    """ Generate a new molecule with the atom order of mol1 and coordinates
      from mol2.
      
      Arguments:
      - mol1:     first instance of the molecule of interest
      - mol2:     second instance the molecule of interest

      Return: RDKit molecule
  """
    match = mol2.GetSubstructMatch(mol1)
    atomNums = tuple(range(mol1.GetNumAtoms()))
    if match != atomNums:
        mol3 = Chem.Mol(mol1)
        mol3.RemoveAllConformers()
        for conf2 in mol2.GetConformers():
            confId = conf2.GetId()
            conf = rdchem.Conformer(mol1.GetNumAtoms())
            conf.SetId(confId)
            for i in range(mol1.GetNumAtoms()):
                conf.SetAtomPosition(i, mol2.GetConformer(confId).GetAtomPosition(match[i]))
            cid = mol3.AddConformer(conf)
        return mol3
    else:
        return Chem.Mol(mol2)