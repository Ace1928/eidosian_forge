import random
import numpy
from rdkit.ML.Neural import ActFuncs, NetNode
def GetNode(self, which):
    """ returns a particular node
    """
    return self.nodeList[which]