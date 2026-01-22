import sys
import weakref
from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions
def GetAllChildren(self):
    """ returns a dictionary, keyed by SMILES, of children """
    res = {}
    for smi, child in self.children.items():
        res[smi] = child
        child._gacRecurse(res, terminalOnly=False)
    return res