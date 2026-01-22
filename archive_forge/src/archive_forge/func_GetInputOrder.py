import pickle
import numpy
from rdkit.ML.Data import DataUtils
def GetInputOrder(self):
    """ returns the input order (used in remapping inputs)

    """
    return self._mapOrder