import random
import numpy
from rdkit.ML.Neural import ActFuncs, NetNode
def GetOutputNodeList(self):
    """ returns a list of output node indices
    """
    return self.layerIndices[-1]