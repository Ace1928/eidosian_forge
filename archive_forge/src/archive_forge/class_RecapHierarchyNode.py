import sys
import weakref
from rdkit import Chem
from rdkit.Chem import rdChemReactions as Reactions
class RecapHierarchyNode(object):
    """ This class is used to hold the Recap hiearchy
    """
    mol = None
    children = None
    parents = None
    smiles = None

    def __init__(self, mol):
        self.mol = mol
        self.children = {}
        self.parents = {}

    def GetAllChildren(self):
        """ returns a dictionary, keyed by SMILES, of children """
        res = {}
        for smi, child in self.children.items():
            res[smi] = child
            child._gacRecurse(res, terminalOnly=False)
        return res

    def GetLeaves(self):
        """ returns a dictionary, keyed by SMILES, of leaf (terminal) nodes """
        res = {}
        for smi, child in self.children.items():
            if not len(child.children):
                res[smi] = child
            else:
                child._gacRecurse(res, terminalOnly=True)
        return res

    def getUltimateParents(self):
        """ returns all the nodes in the hierarchy tree that contain this
            node as a child
        """
        if not self.parents:
            res = [self]
        else:
            res = []
            for p in self.parents.values():
                for uP in p.getUltimateParents():
                    if uP not in res:
                        res.append(uP)
        return res

    def _gacRecurse(self, res, terminalOnly=False):
        for smi, child in self.children.items():
            if not terminalOnly or not len(child.children):
                res[smi] = child
            child._gacRecurse(res, terminalOnly=terminalOnly)

    def __del__(self):
        self.children = {}
        self.parents = {}
        self.mol = None