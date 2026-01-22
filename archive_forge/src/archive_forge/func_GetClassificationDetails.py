import numpy
from rdkit.ML.Data import Quantize
def GetClassificationDetails(self):
    """ returns the probability of the last prediction """
    return self.mprob