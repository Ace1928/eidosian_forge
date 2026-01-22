import random
import numpy
from rdkit.ML.Neural import ActFuncs, NetNode
def GetHiddenLayerNodeList(self, which):
    """ returns a list of hidden nodes in the specified layer
    """
    return self.layerIndices[which + 1]