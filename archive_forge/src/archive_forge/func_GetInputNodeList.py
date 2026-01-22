import random
import numpy
from rdkit.ML.Neural import ActFuncs, NetNode
def GetInputNodeList(self):
    """ returns a list of input node indices
    """
    return self.layerIndices[0]