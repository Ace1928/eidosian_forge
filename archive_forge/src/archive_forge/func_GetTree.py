import pickle
import numpy
from rdkit.ML.DecTree import CrossValidate, PruneTree
def GetTree(self, i):
    return self.treeList[i]